import Foundation

struct RecordingInsights: Codable, Sendable {
    var transcript: String
    var transcriptChunks: [TranscriptChunk]
    var summary: String
    var detailedSummary: String
    var keyPoints: [String]
    var actionItems: [RecordingActionItem]
    var suggestedTitle: String?
    var suggestedTags: [String]
    var category: String?

    private enum CodingKeys: String, CodingKey {
        case transcript
        case transcriptChunks = "transcript_chunks"
        case summary
        case detailedSummary = "detailed_summary"
        case keyPoints = "key_points"
        case actionItems = "action_items"
        case suggestedTitle = "suggested_title"
        case suggestedTags = "suggested_tags"
        case category
    }
}

protocol RecordingIntelligenceProviding: Sendable {
    func analyze(
        recording: Recording,
        audioURL: URL,
        forceRegenerate: Bool
    ) async throws -> RecordingInsights
}

enum IntelligenceError: LocalizedError {
    case invalidServerURL
    case invalidResponse
    case serverUnavailable
    case recordingTooShort
    case server(message: String)

    var errorDescription: String? {
        switch self {
        case .invalidServerURL:
            "The AI server address is invalid."
        case .invalidResponse:
            "The AI server returned an invalid response."
        case .serverUnavailable:
            "The AI server is unavailable. Start the FastAPI server and try again."
        case .recordingTooShort:
            "Recording is too short to summarize."
        case let .server(message):
            message
        }
    }
}
