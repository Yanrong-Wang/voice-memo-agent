import Foundation

struct AgentRecordingContext: Codable, Sendable {
    let id: String
    let title: String
    let createdAt: Date
    let duration: TimeInterval
    let transcript: String?
    let transcriptChunks: [TranscriptChunk]
    let summary: String?
    let keyPoints: [String]
    let actionItems: [RecordingActionItem]
    let suggestedTitle: String?
    let suggestedTags: [String]
    let tags: [String]
    let category: String?

    init(recording: Recording) {
        let hasCurrentAnalysis = recording.transcriptStatus == .ready
            && recording.summaryStatus == .ready
        id = recording.id.uuidString
        title = recording.title
        createdAt = recording.createdAt
        duration = recording.duration
        transcript = hasCurrentAnalysis ? recording.transcript : nil
        transcriptChunks = hasCurrentAnalysis ? recording.transcriptChunks : []
        summary = hasCurrentAnalysis ? recording.summary : nil
        keyPoints = hasCurrentAnalysis ? recording.keyPoints : []
        actionItems = hasCurrentAnalysis ? recording.actionItems : []
        suggestedTitle = hasCurrentAnalysis ? recording.suggestedTitle : nil
        suggestedTags = hasCurrentAnalysis ? recording.suggestedTags : []
        tags = recording.tags
        category = hasCurrentAnalysis ? recording.category : nil
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case title
        case createdAt = "created_at"
        case duration
        case transcript
        case transcriptChunks = "transcript_chunks"
        case summary
        case keyPoints = "key_points"
        case actionItems = "action_items"
        case suggestedTitle = "suggested_title"
        case suggestedTags = "suggested_tags"
        case tags
        case category
    }
}

struct AgentSource: Codable, Hashable, Sendable {
    let recordingID: String
    let recordingTitle: String
    let chunkID: String
    let excerpt: String
    let startTime: TimeInterval?
    let endTime: TimeInterval?
    let score: Double

    private enum CodingKeys: String, CodingKey {
        case recordingID = "recording_id"
        case recordingTitle = "recording_title"
        case chunkID = "chunk_id"
        case excerpt
        case startTime = "start_time"
        case endTime = "end_time"
        case score
    }
}

struct AgentAskRequest: Codable, Sendable {
    let sessionID: String
    let query: String
    let currentRecordingID: String?
    let recordings: [AgentRecordingContext]

    private enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case query
        case currentRecordingID = "current_recording_id"
        case recordings
    }
}

enum AgentActionType: String, Codable, Sendable {
    case renameRecording = "rename_recording"
    case tagRecording = "tag_recording"
    case deleteRecording = "delete_recording"
}

struct AgentPendingAction: Codable, Hashable, Sendable {
    let type: AgentActionType
    let recordingID: String
    let newTitle: String?
    let tags: [String]?
    let recordingTitle: String?

    private enum CodingKeys: String, CodingKey {
        case type
        case recordingID = "recording_id"
        case newTitle = "new_title"
        case tags
        case recordingTitle = "recording_title"
    }
}

struct AgentReply: Codable, Sendable {
    let message: String
    let recordingIDs: [String]
    let sources: [AgentSource]
    let requiresConfirmation: Bool
    let pendingAction: AgentPendingAction?

    private enum CodingKeys: String, CodingKey {
        case message
        case recordingIDs = "recording_ids"
        case sources
        case requiresConfirmation = "requires_confirmation"
        case pendingAction = "pending_action"
    }
}

struct AgentConfirmationRequest: Codable, Sendable {
    let sessionID: String
    let confirm: Bool

    private enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case confirm
    }
}

struct AgentConfirmationReply: Codable, Sendable {
    let success: Bool
    let message: String
    let action: AgentPendingAction?
}

struct AgentMessage: Identifiable, Hashable {
    enum Role {
        case user
        case assistant
    }

    let id = UUID()
    let role: Role
    let text: String
    var recordingIDs: [UUID] = []
    var sources: [AgentSource] = []
    var pendingAction: AgentPendingAction?
}
