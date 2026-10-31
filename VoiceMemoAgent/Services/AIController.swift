import Combine
import Foundation

@MainActor
final class AIController: ObservableObject {
    @Published private(set) var analyzingIDs: Set<UUID> = []
    @Published private(set) var messages: [AgentMessage] = [
        AgentMessage(
            role: .assistant,
            text: "可以让我搜索、总结和整理你的录音。修改标题、标签或删除前，我会先请求确认。"
        )
    ]
    @Published private(set) var isAgentThinking = false
    @Published private(set) var isServerAvailable: Bool?
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
                    text: "这次没有完成：\(message)"
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
                    text: "无法确认操作：\(error.localizedDescription)"
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
    }

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
            "已重命名录音。"
        case .tagRecording:
            "已添加标签。"
        case .deleteRecording:
            "已删除录音。"
        }
    }
}
