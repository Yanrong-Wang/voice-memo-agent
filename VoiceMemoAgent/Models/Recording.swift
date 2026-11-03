import Foundation

enum AIProcessingStatus: String, Codable, Sendable {
    case notStarted = "not_started"
    case processing
    case ready
    case failed
    case expired
}

enum ActionItemPriority: String, Codable, Sendable {
    case low
    case medium
    case high
}

struct RecordingActionItem: Identifiable, Codable, Hashable, Sendable {
    var id = UUID()
    var task: String
    var priority: ActionItemPriority
    var sourceTimestamp: String?

    private enum CodingKeys: String, CodingKey {
        case task
        case priority
        case sourceTimestamp = "source_timestamp"
    }

    init(task: String, priority: ActionItemPriority = .medium, sourceTimestamp: String? = nil) {
        self.task = task
        self.priority = priority
        self.sourceTimestamp = sourceTimestamp
    }
}

struct TranscriptChunk: Identifiable, Codable, Hashable, Sendable {
    var id: String
    var text: String
    var startTime: TimeInterval?
    var endTime: TimeInterval?
    var timingSource: String? = nil

    private enum CodingKeys: String, CodingKey {
        case id
        case text
        case startTime = "start_time"
        case endTime = "end_time"
        case timingSource = "timing_source"
    }
}

struct Recording: Identifiable, Codable, Hashable {
    var id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    var duration: TimeInterval
    var audioFileName: String
    var waveform: [Float]
    var isFavorite: Bool

    var transcriptStatus: AIProcessingStatus
    var summaryStatus: AIProcessingStatus
    var transcript: String?
    var transcriptChunks: [TranscriptChunk]
    var summary: String?
    var detailedSummary: String?
    var keyPoints: [String]
    var actionItems: [RecordingActionItem]
    var suggestedTitle: String?
    var suggestedTags: [String]
    var tags: [String]
    var category: String?
    var analysisError: String?
    var analysisUpdatedAt: Date?

    init(
        id: UUID = UUID(),
        title: String,
        createdAt: Date = .now,
        updatedAt: Date = .now,
        duration: TimeInterval,
        audioFileName: String,
        waveform: [Float] = [],
        isFavorite: Bool = false,
        transcriptStatus: AIProcessingStatus = .notStarted,
        summaryStatus: AIProcessingStatus = .notStarted,
        transcript: String? = nil,
        transcriptChunks: [TranscriptChunk] = [],
        summary: String? = nil,
        detailedSummary: String? = nil,
        keyPoints: [String] = [],
        actionItems: [RecordingActionItem] = [],
        suggestedTitle: String? = nil,
        suggestedTags: [String] = [],
        tags: [String] = [],
        category: String? = nil,
        analysisError: String? = nil,
        analysisUpdatedAt: Date? = nil
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.duration = duration
        self.audioFileName = audioFileName
        self.waveform = waveform
        self.isFavorite = isFavorite
        self.transcriptStatus = transcriptStatus
        self.summaryStatus = summaryStatus
        self.transcript = transcript
        self.transcriptChunks = transcriptChunks
        self.summary = summary
        self.detailedSummary = detailedSummary
        self.keyPoints = keyPoints
        self.actionItems = actionItems
        self.suggestedTitle = suggestedTitle
        self.suggestedTags = suggestedTags
        self.tags = tags
        self.category = category
        self.analysisError = analysisError
        self.analysisUpdatedAt = analysisUpdatedAt
    }

    var searchableText: String {
        [
            title,
            transcript ?? "",
            summary ?? "",
            keyPoints.joined(separator: " "),
            actionItems.map(\.task).joined(separator: " "),
            tags.joined(separator: " "),
            category ?? ""
        ].joined(separator: " ")
    }

    mutating func invalidateAIAnalysis() {
        transcriptStatus = transcript == nil ? .notStarted : .expired
        summaryStatus = summary == nil ? .notStarted : .expired
        analysisError = nil
        updatedAt = .now
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case title
        case createdAt
        case updatedAt
        case duration
        case audioFileName
        case waveform
        case isFavorite
        case transcriptStatus
        case summaryStatus
        case transcript
        case transcriptChunks
        case summary
        case detailedSummary
        case keyPoints
        case actionItems
        case suggestedTitle
        case suggestedTags
        case tags
        case category
        case analysisError
        case analysisUpdatedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        title = try container.decode(String.self, forKey: .title)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? createdAt
        duration = try container.decode(TimeInterval.self, forKey: .duration)
        audioFileName = try container.decode(String.self, forKey: .audioFileName)
        waveform = try container.decodeIfPresent([Float].self, forKey: .waveform) ?? []
        isFavorite = try container.decodeIfPresent(Bool.self, forKey: .isFavorite) ?? false
        transcript = try container.decodeIfPresent(String.self, forKey: .transcript)
        summary = try container.decodeIfPresent(String.self, forKey: .summary)
        transcriptStatus = try container.decodeIfPresent(
            AIProcessingStatus.self,
            forKey: .transcriptStatus
        ) ?? (transcript == nil ? .notStarted : .ready)
        summaryStatus = try container.decodeIfPresent(
            AIProcessingStatus.self,
            forKey: .summaryStatus
        ) ?? (summary == nil ? .notStarted : .ready)
        transcriptChunks = try container.decodeIfPresent(
            [TranscriptChunk].self,
            forKey: .transcriptChunks
        ) ?? []
        detailedSummary = try container.decodeIfPresent(String.self, forKey: .detailedSummary)
        keyPoints = try container.decodeIfPresent([String].self, forKey: .keyPoints) ?? []
        actionItems = try container.decodeIfPresent(
            [RecordingActionItem].self,
            forKey: .actionItems
        ) ?? []
        suggestedTitle = try container.decodeIfPresent(String.self, forKey: .suggestedTitle)
        suggestedTags = try container.decodeIfPresent([String].self, forKey: .suggestedTags) ?? []
        tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
        category = try container.decodeIfPresent(String.self, forKey: .category)
        analysisError = try container.decodeIfPresent(String.self, forKey: .analysisError)
        analysisUpdatedAt = try container.decodeIfPresent(Date.self, forKey: .analysisUpdatedAt)
    }
}

extension Recording {
    static func suggestedTitle(on date: Date = .now) -> String {
        let formatter = DateFormatter()
        formatter.locale = .current
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return "New Recording \(formatter.string(from: date))"
    }
}
