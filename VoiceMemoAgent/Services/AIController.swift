import Combine
import Foundation

@MainActor
final class AIController: ObservableObject {
    @Published private(set) var analyzingIDs: Set<UUID> = []
    @Published private(set) var messages: [AgentMessage] = [
        AgentMessage(
            role: .assistant,
            text: "I can search, summarize, and organize your recordings. "
                + "I will ask for confirmation before renaming, tagging, or deleting anything."
        )
    ]
    @Published private(set) var isAgentThinking = false
    @Published private(set) var isServerAvailable: Bool?
    @Published private(set) var isUsingLocalDemo = false
    @Published var lastError: String?

    private let client: AIAPIClient
    private let sessionID: String

    init(client: AIAPIClient = AIAPIClient()) {
        self.client = client
        let key = "ai.agentSessionID"
        if let existing = UserDefaults.standard.string(forKey: key) {
            sessionID = existing
        } else {
            let newID = UUID().uuidString
            UserDefaults.standard.set(newID, forKey: key)
            sessionID = newID
        }
    }

    func analyze(
        recordingID: UUID,
        in store: RecordingStore,
        forceRegenerate: Bool = false
    ) async {
        guard
            !analyzingIDs.contains(recordingID),
            let recording = store.recordings.first(where: { $0.id == recordingID })
        else {
            return
        }
        if recording.summaryStatus == .ready && !forceRegenerate {
            return
        }
        guard recording.duration >= 3 else {
            store.markAnalysisFailed(
                recordingID,
                message: IntelligenceError.recordingTooShort.localizedDescription
            )
            return
        }

        analyzingIDs.insert(recordingID)
        store.markAnalysisProcessing(recordingID)
        defer { analyzingIDs.remove(recordingID) }

        do {
            let insights = try await client.analyze(
                recording: recording,
                audioURL: store.audioURL(for: recording),
                forceRegenerate: forceRegenerate
            )
            store.updateInsights(insights, for: recordingID)
        } catch {
            let message = error.localizedDescription
            store.markAnalysisFailed(recordingID, message: message)
            lastError = message
        }
    }

    func ask(
        _ query: String,
        currentRecordingID: UUID? = nil,
        store: RecordingStore
    ) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isAgentThinking else { return }

        messages.append(AgentMessage(role: .user, text: trimmed))
        isAgentThinking = true
        defer { isAgentThinking = false }

#if DEBUG
#if targetEnvironment(simulator)
        if isUsingLocalDemo {
            messages.append(
                localDemoReply(
                    to: trimmed,
                    currentRecordingID: currentRecordingID,
                    recordings: store.recordings
                )
            )
            return
        }
#endif
#endif

        do {
            let reply = try await client.ask(
                AgentAskRequest(
                    sessionID: sessionID,
                    query: trimmed,
                    currentRecordingID: currentRecordingID?.uuidString,
                    recordings: store.recordings.map(AgentRecordingContext.init)
                )
            )
            let ids = reply.recordingIDs.compactMap(UUID.init(uuidString:))
            messages.append(
                AgentMessage(
                    role: .assistant,
                    text: reply.message,
                    recordingIDs: ids,
                    sources: reply.sources,
                    pendingAction: reply.pendingAction
                )
            )
        } catch {
            let message = error.localizedDescription
            lastError = message
            messages.append(
                AgentMessage(
                    role: .assistant,
                    text: "I couldn't complete that request: \(message)"
                )
            )
        }
    }

    func confirmPendingAction(
        _ action: AgentPendingAction,
        confirm: Bool,
        store: RecordingStore
    ) async {
        guard !isAgentThinking else { return }
        isAgentThinking = true
        defer { isAgentThinking = false }

        do {
            let reply = try await client.confirm(
                AgentConfirmationRequest(sessionID: sessionID, confirm: confirm)
            )
            if confirm, let confirmedAction = reply.action {
                apply(confirmedAction, to: store)
            }
            if let index = messages.lastIndex(where: { $0.pendingAction == action }) {
                messages[index].pendingAction = nil
            }
            messages.append(
                AgentMessage(
                    role: .assistant,
                    text: confirm && reply.success ? completionMessage(for: action) : reply.message
                )
            )
        } catch {
            lastError = error.localizedDescription
            messages.append(
                AgentMessage(
                    role: .assistant,
                    text: "I couldn't confirm the action: \(error.localizedDescription)"
                )
            )
        }
    }

    func checkServer() async {
        do {
            isServerAvailable = try await client.healthCheck()
        } catch {
            isServerAvailable = false
        }
#if DEBUG
#if targetEnvironment(simulator)
        isUsingLocalDemo = isServerAvailable != true
#else
        isUsingLocalDemo = false
#endif
#else
        isUsingLocalDemo = false
#endif
    }

#if DEBUG
#if targetEnvironment(simulator)
    private func localDemoReply(
        to query: String,
        currentRecordingID: UUID?,
        recordings: [Recording]
    ) -> AgentMessage {
        let normalizedQuery = query.lowercased()
        let queryTokens = searchTokens(in: normalizedQuery)
        let asksForActions = normalizedQuery.contains("action")
            || normalizedQuery.contains("todo")
            || normalizedQuery.contains("follow up")

        struct Match {
            let recording: Recording
            let chunk: TranscriptChunk
            let score: Int
        }

        var matches: [Match] = []
        for recording in recordings where recording.transcriptStatus == .ready {
            let titleAndTags = ([recording.title] + recording.tags)
                .joined(separator: " ")
                .lowercased()
            for chunk in recording.transcriptChunks {
                let chunkText = chunk.text.lowercased()
                var score = queryTokens.reduce(into: 0) { result, token in
                    if titleAndTags.contains(token) { result += 3 }
                    if chunkText.contains(token) { result += 2 }
                }
                if recording.id == currentRecordingID { score += 1 }
                if asksForActions, !recording.actionItems.isEmpty { score += 2 }
                if score > 0 {
                    matches.append(Match(recording: recording, chunk: chunk, score: score))
                }
            }
        }

        let bestMatches = matches
            .sorted { lhs, rhs in
                lhs.score == rhs.score
                    ? lhs.recording.createdAt > rhs.recording.createdAt
                    : lhs.score > rhs.score
            }
            .reduce(into: [Match]()) { result, match in
                guard !result.contains(where: { $0.recording.id == match.recording.id }) else {
                    return
                }
                if result.count < 3 { result.append(match) }
            }

        guard !bestMatches.isEmpty else {
            return AgentMessage(
                role: .assistant,
                text: "Local demo search could not find a matching transcript. Try asking about RAG, interview evaluation, product action items, or React Native migration."
            )
        }

        let sources = bestMatches.enumerated().map { index, match in
            AgentSource(
                recordingID: match.recording.id.uuidString,
                recordingTitle: match.recording.title,
                chunkID: match.chunk.id,
                excerpt: match.chunk.text,
                startTime: match.chunk.startTime,
                endTime: match.chunk.endTime,
                score: max(0.1, min(0.99, Double(match.score) / 12.0))
            )
        }
        let primary = bestMatches[0].recording
        let response: String
        if asksForActions, !primary.actionItems.isEmpty {
            let tasks = primary.actionItems.map(\.task).joined(separator: " ")
            response = "The main follow-up is: \(tasks) [S1]"
        } else {
            response = "\(primary.summary ?? bestMatches[0].chunk.text) [S1]"
        }

        return AgentMessage(
            role: .assistant,
            text: response,
            recordingIDs: bestMatches.map(\.recording.id),
            sources: sources
        )
    }

    private func searchTokens(in text: String) -> Set<String> {
        let stopWords: Set<String> = [
            "a", "about", "and", "are", "did", "do", "for", "i", "in", "is",
            "me", "my", "of", "on", "the", "to", "what", "which", "with"
        ]
        let words = text.split { !$0.isLetter && !$0.isNumber }.map(String.init)
        var tokens = Set(words.filter { $0.count > 1 && !stopWords.contains($0) })

        if tokens.contains("retrieval") || tokens.contains("embedding") {
            tokens.insert("rag")
            tokens.insert("vector")
        }
        if tokens.contains("tasks") || tokens.contains("todo") {
            tokens.insert("action")
        }
        return tokens
    }
#endif
#endif

    private func apply(_ action: AgentPendingAction, to store: RecordingStore) {
        guard let id = UUID(uuidString: action.recordingID) else { return }
        switch action.type {
        case .renameRecording:
            if let title = action.newTitle {
                store.rename(id, to: title)
            }
        case .tagRecording:
            let current = store.recordings.first(where: { $0.id == id })?.tags ?? []
            store.setTags(current + (action.tags ?? []), for: id)
        case .deleteRecording:
            store.delete(id)
        }
    }

    private func completionMessage(for action: AgentPendingAction) -> String {
        switch action.type {
        case .renameRecording:
            "The recording was renamed."
        case .tagRecording:
            "The tags were added."
        case .deleteRecording:
            "The recording was deleted."
        }
    }
}
