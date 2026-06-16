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
