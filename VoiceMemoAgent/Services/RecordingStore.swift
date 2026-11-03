import AVFoundation
import Combine
import Foundation

@MainActor
final class RecordingStore: ObservableObject {
    @Published private(set) var recordings: [Recording] = []
    @Published var lastError: String?

    private let fileManager = FileManager.default
    private let metadataURL: URL

    static let documentsURL: URL = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }()

    static let audioDirectoryURL: URL = {
        documentsURL.appendingPathComponent("Recordings", isDirectory: true)
    }()

    init() {
        metadataURL = Self.documentsURL.appendingPathComponent("recordings.json")
        createDirectories()
        load()
#if DEBUG
#if targetEnvironment(simulator)
        seedDemoRecordingsIfNeeded()
#endif
#endif
    }

    func audioURL(for recording: Recording) -> URL {
        Self.audioDirectoryURL.appendingPathComponent(recording.audioFileName)
    }

    func addRecording(fileURL: URL, duration: TimeInterval, liveWaveform: [Float]) async -> UUID {
        let id = UUID()
        let fileName = "\(id.uuidString).m4a"
        let destination = Self.audioDirectoryURL.appendingPathComponent(fileName)

        do {
            if fileURL != destination {
                if fileManager.fileExists(atPath: destination.path) {
                    try fileManager.removeItem(at: destination)
                }
                try fileManager.moveItem(at: fileURL, to: destination)
            }

            let sampled = await WaveformSampler.samples(from: destination, count: 180)
            let recording = Recording(
                id: id,
                title: Recording.suggestedTitle(),
                duration: duration,
                audioFileName: fileName,
                waveform: sampled.isEmpty ? liveWaveform : sampled
            )
            recordings.insert(recording, at: 0)
            save()
        } catch {
            lastError = error.localizedDescription
        }
        return id
    }

    func rename(_ id: UUID, to newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = index(for: id) else { return }
        recordings[index].title = trimmed
        recordings[index].updatedAt = .now
        save()
    }

    func setTags(_ tags: [String], for id: UUID) {
        guard let index = index(for: id) else { return }
        recordings[index].tags = Array(
            Set(tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) })
        )
        .filter { !$0.isEmpty }
        .sorted()
        recordings[index].updatedAt = .now
        save()
    }

    func toggleFavorite(_ id: UUID) {
        guard let index = index(for: id) else { return }
        recordings[index].isFavorite.toggle()
        recordings[index].updatedAt = .now
        save()
    }

    func markAnalysisProcessing(_ id: UUID) {
        guard let index = index(for: id) else { return }
        recordings[index].transcriptStatus = .processing
        recordings[index].summaryStatus = .processing
        recordings[index].analysisError = nil
        save()
    }

    func markAnalysisFailed(_ id: UUID, message: String) {
        guard let index = index(for: id) else { return }
        recordings[index].transcriptStatus = .failed
        recordings[index].summaryStatus = .failed
        recordings[index].analysisError = message
        save()
    }

    func updateInsights(_ insights: RecordingInsights, for id: UUID) {
        guard let index = index(for: id) else { return }
        recordings[index].transcript = insights.transcript
        recordings[index].transcriptChunks = insights.transcriptChunks
        recordings[index].summary = insights.summary
        recordings[index].detailedSummary = insights.detailedSummary
        recordings[index].keyPoints = insights.keyPoints
        recordings[index].actionItems = insights.actionItems
        recordings[index].suggestedTitle = insights.suggestedTitle
        recordings[index].suggestedTags = insights.suggestedTags
        recordings[index].category = insights.category
        recordings[index].transcriptStatus = .ready
        recordings[index].summaryStatus = .ready
        recordings[index].analysisError = nil
        recordings[index].analysisUpdatedAt = .now
        recordings[index].updatedAt = .now
        save()
    }

    func delete(_ id: UUID) {
        guard let index = index(for: id) else { return }
        let recording = recordings.remove(at: index)
        try? fileManager.removeItem(at: audioURL(for: recording))
        save()
    }

    func duplicate(_ id: UUID) {
        guard let source = recordings.first(where: { $0.id == id }) else { return }
        let duplicateID = UUID()
        let fileName = "\(duplicateID.uuidString).m4a"

        do {
            try fileManager.copyItem(
                at: audioURL(for: source),
                to: Self.audioDirectoryURL.appendingPathComponent(fileName)
            )
            var copy = source
            copy.id = duplicateID
            copy.title = "\(source.title) copy"
            copy.createdAt = .now
            copy.updatedAt = .now
            copy.audioFileName = fileName
            recordings.insert(copy, at: 0)
            save()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func trim(_ id: UUID, start: TimeInterval, end: TimeInterval) async -> Bool {
        guard let index = index(for: id), end > start else { return false }
        let recording = recordings[index]
        let sourceURL = audioURL(for: recording)
        let exportURL = Self.audioDirectoryURL.appendingPathComponent("\(UUID().uuidString)-trim.m4a")
        let asset = AVURLAsset(url: sourceURL)

        guard let exporter = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            lastError = "This recording cannot be trimmed."
            return false
        }

        exporter.outputURL = exportURL
        exporter.outputFileType = .m4a
        exporter.timeRange = CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 600),
            duration: CMTime(seconds: end - start, preferredTimescale: 600)
        )

        await withCheckedContinuation { continuation in
            exporter.exportAsynchronously {
                continuation.resume()
            }
        }

        guard exporter.status == .completed else {
            lastError = exporter.error?.localizedDescription ?? "Trim failed."
            try? fileManager.removeItem(at: exportURL)
            return false
        }

        do {
            try fileManager.removeItem(at: sourceURL)
            try fileManager.moveItem(at: exportURL, to: sourceURL)
            recordings[index].duration = end - start
            recordings[index].waveform = await WaveformSampler.samples(from: sourceURL, count: 180)
            recordings[index].invalidateAIAnalysis()
            save()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    func deleteSegment(_ id: UUID, start: TimeInterval, end: TimeInterval) async -> Bool {
        guard
            let index = index(for: id),
            end > start,
            start > 0 || end < recordings[index].duration
        else {
            return false
        }

        let recording = recordings[index]
        let sourceURL = audioURL(for: recording)
        let asset = AVURLAsset(url: sourceURL)
        guard let sourceTrack = try? await asset.loadTracks(withMediaType: .audio).first else {
            lastError = "The audio track could not be read."
            return false
        }

        let composition = AVMutableComposition()
        guard let destinationTrack = composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            lastError = "The edit could not be prepared."
            return false
        }

        do {
            let startTime = CMTime(seconds: start, preferredTimescale: 600)
            let endTime = CMTime(seconds: end, preferredTimescale: 600)
            let assetDuration = try await asset.load(.duration)

            if start > 0 {
                try destinationTrack.insertTimeRange(
                    CMTimeRange(start: .zero, end: startTime),
                    of: sourceTrack,
                    at: .zero
                )
            }
            if end < recording.duration {
                try destinationTrack.insertTimeRange(
                    CMTimeRange(start: endTime, end: assetDuration),
                    of: sourceTrack,
                    at: startTime
                )
            }
        } catch {
            lastError = error.localizedDescription
            return false
        }

        let exportURL = Self.audioDirectoryURL.appendingPathComponent("\(UUID().uuidString)-edit.m4a")
        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            lastError = "This recording cannot be edited."
            return false
        }
        exporter.outputURL = exportURL
        exporter.outputFileType = .m4a

        await withCheckedContinuation { continuation in
            exporter.exportAsynchronously {
                continuation.resume()
            }
        }

        guard exporter.status == .completed else {
            lastError = exporter.error?.localizedDescription ?? "Edit failed."
            try? fileManager.removeItem(at: exportURL)
            return false
        }

        do {
            try fileManager.removeItem(at: sourceURL)
            try fileManager.moveItem(at: exportURL, to: sourceURL)
            recordings[index].duration = max(0, recording.duration - (end - start))
            recordings[index].waveform = await WaveformSampler.samples(from: sourceURL, count: 180)
            recordings[index].invalidateAIAnalysis()
            save()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    private func index(for id: UUID) -> Int? {
        recordings.firstIndex(where: { $0.id == id })
    }

    private func createDirectories() {
        do {
            try fileManager.createDirectory(
                at: Self.audioDirectoryURL,
                withIntermediateDirectories: true
            )
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func load() {
        guard fileManager.fileExists(atPath: metadataURL.path) else { return }
        do {
            let data = try Data(contentsOf: metadataURL)
            recordings = try JSONDecoder().decode([Recording].self, from: data)
                .filter { fileManager.fileExists(atPath: audioURL(for: $0).path) }
                .sorted { $0.createdAt > $1.createdAt }
        } catch {
            lastError = error.localizedDescription
        }
    }

#if DEBUG
#if targetEnvironment(simulator)
    private func seedDemoRecordingsIfNeeded() {
        guard recordings.isEmpty else { return }

        do {
            let now = Date()
            recordings = [
                try makeDemoRecording(
                    id: UUID(uuidString: "A1000000-0000-0000-0000-000000000001")!,
                    resourceName: "ai-engineer-interview-review",
                    title: "AI Engineer Interview Review",
                    createdAt: Calendar.current.date(byAdding: .hour, value: -2, to: now) ?? now,
                    duration: 30.87,
                    transcript: "In the AI engineer interview, I introduced a voice memo assistant built with SwiftUI, FastAPI, LangChain, and PostgreSQL with pgvector. We discussed retrieval augmented generation, splitting transcripts into timestamped chunks, embedding each chunk, and returning sources that jump to the exact moment in a recording. The interviewer asked how I evaluate retrieval quality. I proposed recall at K, citation accuracy, unsupported claim rate, and a small set of human reviewed questions. My follow up is to add hybrid search and compare it with vector search.",
                    chunks: [
                        TranscriptChunk(
                            id: "interview_intro",
                            text: "I introduced a voice memo assistant built with SwiftUI, FastAPI, LangChain, and PostgreSQL with pgvector.",
                            startTime: 0,
                            endTime: 8.4
                        ),
                        TranscriptChunk(
                            id: "rag_design",
                            text: "We discussed RAG, timestamped transcript chunks, embeddings, and source cards that jump to the exact audio moment.",
                            startTime: 8.4,
                            endTime: 19.2
                        ),
                        TranscriptChunk(
                            id: "evaluation",
                            text: "I proposed recall at K, citation accuracy, unsupported claim rate, human review, and a hybrid search experiment.",
                            startTime: 19.2,
                            endTime: 30.87
                        )
                    ],
                    summary: "Reviewed an AI engineer interview focused on the Voice Memo Agent's RAG architecture and retrieval evaluation strategy.",
                    detailedSummary: "The discussion covered the SwiftUI and FastAPI architecture, LangChain transcript chunking, pgvector similarity search, and timestamp-linked citations. The evaluation plan combines retrieval recall, citation accuracy, unsupported-claim tracking, and human-reviewed questions. The next experiment is a comparison between vector-only and hybrid retrieval.",
                    keyPoints: [
                        "Timestamped chunks make retrieved evidence directly playable.",
                        "pgvector stores embeddings and recording metadata.",
                        "Evaluation should measure retrieval and answer grounding separately."
                    ],
                    actionItems: [
                        RecordingActionItem(
                            task: "Add a hybrid-search experiment and compare recall at K.",
                            priority: .high,
                            sourceTimestamp: "00:25"
                        )
                    ],
                    tags: ["AI", "Interview", "RAG"],
                    category: "Career"
                ),
                try makeDemoRecording(
                    id: UUID(uuidString: "A1000000-0000-0000-0000-000000000002")!,
                    resourceName: "product-sync-notes",
                    title: "Product Sync Notes",
                    createdAt: Calendar.current.date(byAdding: .day, value: -1, to: now) ?? now,
                    duration: 15.15,
                    transcript: "Product sync notes. The team agreed to finish the onboarding prototype by Friday. Maya will update the empty state and Daniel will verify analytics events. We should test the new recording flow with five users and collect feedback about transcription accuracy. The next review is Monday at ten in the morning.",
                    chunks: [
                        TranscriptChunk(
                            id: "product_decisions",
                            text: "Finish the onboarding prototype by Friday. Maya owns the empty state and Daniel owns analytics verification.",
                            startTime: 0,
                            endTime: 8.1
                        ),
                        TranscriptChunk(
                            id: "product_follow_up",
                            text: "Test the recording flow with five users and review transcription feedback on Monday at ten.",
                            startTime: 8.1,
                            endTime: 15.15
                        )
                    ],
                    summary: "The team assigned onboarding work, planned a five-user recording test, and scheduled a Monday review.",
                    detailedSummary: "The onboarding prototype is due Friday. Maya will update the empty state, Daniel will verify analytics events, and the team will test the recording flow with five users before the next review on Monday at 10:00.",
                    keyPoints: [
                        "Onboarding prototype is due Friday.",
                        "Five users will test the new recording flow.",
                        "The next review is Monday at 10:00."
                    ],
                    actionItems: [
                        RecordingActionItem(task: "Update the onboarding empty state.", priority: .high, sourceTimestamp: "00:04"),
                        RecordingActionItem(task: "Verify analytics events.", priority: .medium, sourceTimestamp: "00:06")
                    ],
                    tags: ["Meeting", "Product"],
                    category: "Work"
                ),
                try makeDemoRecording(
                    id: UUID(uuidString: "A1000000-0000-0000-0000-000000000003")!,
                    resourceName: "react-native-migration",
                    title: "React Native Migration Plan",
                    createdAt: Calendar.current.date(byAdding: .day, value: -3, to: now) ?? now,
                    duration: 16.76,
                    transcript: "React Native migration plan. Start with the settings and profile screens because they have few native dependencies. Keep audio recording and waveform rendering in Swift until the bridge performance is measured. Add end to end tests before moving navigation, and use a staged rollout so the existing iPhone application remains available.",
                    chunks: [
                        TranscriptChunk(
                            id: "migration_scope",
                            text: "Start the React Native migration with settings and profile because they have few native dependencies.",
                            startTime: 0,
                            endTime: 6.2
                        ),
                        TranscriptChunk(
                            id: "native_audio",
                            text: "Keep recording and waveform rendering in Swift until bridge performance is measured.",
                            startTime: 6.2,
                            endTime: 11.4
                        ),
                        TranscriptChunk(
                            id: "migration_rollout",
                            text: "Add end-to-end tests before moving navigation and use a staged rollout.",
                            startTime: 11.4,
                            endTime: 16.76
                        )
                    ],
                    summary: "A staged React Native migration should begin with low-risk screens while audio remains native Swift.",
                    detailedSummary: "Settings and profile are the first migration candidates. Recording and waveform code remain in Swift until bridge performance is proven. End-to-end coverage and a staged rollout reduce migration risk.",
                    keyPoints: [
                        "Begin with settings and profile.",
                        "Keep latency-sensitive audio code in Swift initially.",
                        "Add end-to-end tests before migrating navigation."
                    ],
                    actionItems: [
                        RecordingActionItem(task: "Benchmark audio bridge performance.", priority: .high, sourceTimestamp: "00:08")
                    ],
                    tags: ["Architecture", "React Native"],
                    category: "Engineering"
                )
            ]
            save()
        } catch {
            lastError = "Could not create simulator demo recordings: \(error.localizedDescription)"
        }
    }

    private func makeDemoRecording(
        id: UUID,
        resourceName: String,
        title: String,
        createdAt: Date,
        duration: TimeInterval,
        transcript: String,
        chunks: [TranscriptChunk],
        summary: String,
        detailedSummary: String,
        keyPoints: [String],
        actionItems: [RecordingActionItem],
        tags: [String],
        category: String
    ) throws -> Recording {
        guard let sourceURL = Bundle.main.url(forResource: resourceName, withExtension: "m4a") else {
            throw CocoaError(.fileNoSuchFile)
        }

        let fileName = "\(id.uuidString).m4a"
        let destination = Self.audioDirectoryURL.appendingPathComponent(fileName)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: sourceURL, to: destination)

        return Recording(
            id: id,
            title: title,
            createdAt: createdAt,
            updatedAt: createdAt,
            duration: duration,
            audioFileName: fileName,
            waveform: demoWaveform(seed: resourceName.hashValue),
            transcriptStatus: .ready,
            summaryStatus: .ready,
            transcript: transcript,
            transcriptChunks: chunks,
            summary: summary,
            detailedSummary: detailedSummary,
            keyPoints: keyPoints,
            actionItems: actionItems,
            suggestedTitle: title,
            suggestedTags: tags,
            tags: tags,
            category: category,
            analysisUpdatedAt: createdAt
        )
    }

    private func demoWaveform(seed: Int) -> [Float] {
        (0..<180).map { index in
            let primary = abs(sin(Double(index + abs(seed % 19)) * 0.19))
            let detail = abs(sin(Double(index) * 0.61)) * 0.18
            return Float(min(0.94, 0.08 + primary * 0.58 + detail))
        }
    }
#endif
#endif

    private func save() {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(recordings).write(to: metadataURL, options: .atomic)
        } catch {
            lastError = error.localizedDescription
        }
    }
}
