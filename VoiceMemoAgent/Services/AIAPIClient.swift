import Foundation

enum AIConfiguration {
    static let serverURLKey = "ai.serverURL"
    static let apiTokenKey = "ai.apiToken"
    static let defaultServerURL = "http://127.0.0.1:8000"

    static var serverURLString: String {
        UserDefaults.standard.string(forKey: serverURLKey) ?? defaultServerURL
    }

    static var apiToken: String? {
        let value = UserDefaults.standard.string(forKey: apiTokenKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value?.isEmpty == false ? value : nil
    }
}

struct AIAPIClient: RecordingIntelligenceProviding, Sendable {
    private struct ServerError: Decodable {
        let detail: String
    }

    private struct AnalysisJobSubmission: Decodable {
        let jobID: String

        private enum CodingKeys: String, CodingKey {
            case jobID = "job_id"
        }
    }

    private struct AnalysisJobStatus: Decodable {
        let status: String
        let result: RecordingInsights?
        let error: String?
    }

    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func analyze(
        recording: Recording,
        audioURL: URL,
        forceRegenerate: Bool
    ) async throws -> RecordingInsights {
        let endpoint = try makeURL(
            path: "/recordings/\(recording.id.uuidString)/analysis-jobs"
        )
        let boundary = "VoiceMemoBoundary-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = try multipartBody(
            boundary: boundary,
            audioURL: audioURL,
            duration: recording.duration,
            forceRegenerate: forceRegenerate
        )

        let data = try await perform(request)
        let submission = try decoder.decode(AnalysisJobSubmission.self, from: data)

        for _ in 0..<300 {
            try await Task.sleep(for: .seconds(1))
            var statusRequest = URLRequest(
                url: try makeURL(path: "/analysis-jobs/\(submission.jobID)")
            )
            statusRequest.timeoutInterval = 15
            let statusData = try await perform(statusRequest)
            let job = try decoder.decode(AnalysisJobStatus.self, from: statusData)
            if job.status == "finished", let result = job.result {
                return result
            }
            if ["failed", "stopped", "canceled"].contains(job.status) {
                throw IntelligenceError.server(
                    message: job.error ?? "Audio analysis failed."
                )
            }
        }
        throw IntelligenceError.server(message: "Audio analysis timed out.")
    }

    func ask(_ requestBody: AgentAskRequest) async throws -> AgentReply {
        try await sendJSON(path: "/agent/ask", body: requestBody)
    }

    func confirm(_ requestBody: AgentConfirmationRequest) async throws -> AgentConfirmationReply {
        try await sendJSON(path: "/agent/confirm", body: requestBody)
    }

    func healthCheck() async throws -> Bool {
        var request = URLRequest(url: try makeURL(path: "/health"))
        request.timeoutInterval = 8
        let data = try await perform(request)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return object?["status"] as? String == "ok"
    }

    private func sendJSON<Request: Encodable, Response: Decodable>(
        path: String,
        body: Request
    ) async throws -> Response {
        var request = URLRequest(url: try makeURL(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        let data = try await perform(request)
        return try decoder.decode(Response.self, from: data)
    }

    private func perform(_ originalRequest: URLRequest) async throws -> Data {
        var request = originalRequest
        if let token = AIConfiguration.apiToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw IntelligenceError.invalidResponse
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                let message = (try? decoder.decode(ServerError.self, from: data).detail)
                    ?? "AI request failed with status \(httpResponse.statusCode)."
                throw IntelligenceError.server(message: message)
            }
            return data
        } catch let error as IntelligenceError {
            throw error
        } catch {
            let code = (error as? URLError)?.code
            if code == .cannotConnectToHost || code == .cannotFindHost || code == .timedOut {
                throw IntelligenceError.serverUnavailable
            }
            throw error
        }
    }

    private func makeURL(path: String) throws -> URL {
        let value = AIConfiguration.serverURLString
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: "\(value)\(path)") else {
            throw IntelligenceError.invalidServerURL
        }
        return url
    }

    private func multipartBody(
        boundary: String,
        audioURL: URL,
        duration: TimeInterval,
        forceRegenerate: Bool
    ) throws -> Data {
        var body = Data()

        func append(_ string: String) {
            body.append(Data(string.utf8))
        }

        func appendField(name: String, value: String) {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            append("\(value)\r\n")
        }

        appendField(name: "duration", value: String(duration))
        appendField(name: "force_regenerate", value: forceRegenerate ? "true" : "false")
        append("--\(boundary)\r\n")
        append(
            "Content-Disposition: form-data; name=\"audio\"; "
                + "filename=\"\(audioURL.lastPathComponent)\"\r\n"
        )
        append("Content-Type: audio/mp4\r\n\r\n")
        body.append(try Data(contentsOf: audioURL))
        append("\r\n--\(boundary)--\r\n")
        return body
    }

    private var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
