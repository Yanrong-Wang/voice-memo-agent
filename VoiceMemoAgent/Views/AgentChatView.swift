import SwiftUI

struct AgentChatView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    let currentRecordingID: UUID?

    @State private var query = ""
    @State private var showsSettings = false
    @State private var selectedRecordingID: UUID?
    @State private var selectedStartTime: TimeInterval?

    init(currentRecordingID: UUID? = nil) {
        self.currentRecordingID = currentRecordingID
    }

    var body: some View {
        NavigationStack {
            ZStack {
                MemoTheme.background.ignoresSafeArea()
                VStack(spacing: 0) {
                    messages
                    inputBar
                }
            }
            .foregroundStyle(.white)
            .navigationTitle("Memo Agent")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showsSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
        }
        .sheet(isPresented: $showsSettings) {
            AISettingsView()
                .environmentObject(ai)
        }
        .fullScreenCover(
            isPresented: Binding(
                get: { selectedRecordingID != nil },
                set: { if !$0 { selectedRecordingID = nil } }
            )
        ) {
            if let selectedRecordingID {
                RecordingDetailView(
                    recordingID: selectedRecordingID,
                    initialPlaybackTime: selectedStartTime
                )
                    .environmentObject(store)
                    .environmentObject(audio)
                    .environmentObject(ai)
            }
        }
        .task {
            await ai.checkServer()
        }
    }

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    serverStatus
                    ForEach(ai.messages) { message in
                        messageView(message)
                            .id(message.id)
                    }
                    if ai.isAgentThinking {
                        HStack {
                            ProgressView()
                                .tint(.white)
                            Text("Agent is thinking...")
                                .font(.subheadline)
                                .foregroundStyle(MemoTheme.secondary)
                            Spacer()
                        }
                        .padding(.horizontal, 18)
                    }
                }
                .padding(.vertical, 14)
            }
            .onChange(of: ai.messages.count) { _, _ in
                if let id = ai.messages.last?.id {
                    withAnimation { proxy.scrollTo(id, anchor: .bottom) }
                }
            }
        }
    }

    @ViewBuilder
    private var serverStatus: some View {
        if ai.isUsingLocalDemo {
            Label(
                "Local Demo Mode: source-linked search uses the bundled transcripts.",
                systemImage: "laptopcomputer.and.iphone"
            )
            .font(.caption.weight(.semibold))
            .foregroundStyle(MemoTheme.waveform)
            .padding(10)
            .frame(maxWidth: .infinity)
            .background(MemoTheme.waveform.opacity(0.12))
            .padding(.horizontal, 18)
        } else if ai.isServerAvailable == false {
            Button {
                showsSettings = true
            } label: {
                Label(
                    "AI server is offline. Tap to configure.",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption.weight(.semibold))
                .foregroundStyle(.yellow)
                .padding(10)
                .frame(maxWidth: .infinity)
                .background(Color.yellow.opacity(0.12))
            }
            .padding(.horizontal, 18)
        }
    }

    private func messageView(_ message: AgentMessage) -> some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 10) {
            HStack {
                if message.role == .user { Spacer(minLength: 54) }
                Text(message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                    .background(
                        RoundedRectangle(cornerRadius: 17)
                            .fill(
                                message.role == .user
                                    ? MemoTheme.waveform
                                    : Color.white.opacity(0.1)
                            )
                    )
                if message.role == .assistant { Spacer(minLength: 54) }
            }

            if message.sources.isEmpty {
                ForEach(message.recordingIDs, id: \.self) { id in
                    if let recording = store.recordings.first(where: { $0.id == id }) {
                        recordingCard(recording)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Sources")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(MemoTheme.secondary)
                    ForEach(Array(message.sources.enumerated()), id: \.element) { index, source in
                        sourceCard(source, label: "S\(index + 1)")
                    }
                }
            }

            if let action = message.pendingAction {
                confirmationCard(action)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 18)
    }

    private func recordingCard(_ recording: Recording) -> some View {
        Button {
            selectedStartTime = nil
            selectedRecordingID = recording.id
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "waveform")
                    .foregroundStyle(MemoTheme.waveform)
                VStack(alignment: .leading, spacing: 4) {
                    Text(recording.title)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    if let summary = recording.summary {
                        Text(summary)
                            .font(.caption)
                            .foregroundStyle(MemoTheme.secondary)
                            .lineLimit(2)
                    }
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(MemoTheme.secondary)
            }
            .padding(13)
            .background(
                RoundedRectangle(cornerRadius: 15)
                    .fill(Color.white.opacity(0.07))
            )
        }
        .buttonStyle(.plain)
    }

    private func sourceCard(_ source: AgentSource, label: String) -> some View {
        Button {
            selectedStartTime = source.startTime
            selectedRecordingID = UUID(uuidString: source.recordingID)
        } label: {
            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text("[\(label)] \(source.recordingTitle)")
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    Spacer()
                    if let startTime = source.startTime {
                        Text(sourceTime(startTime, source.endTime))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(MemoTheme.waveform)
                    }
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(MemoTheme.secondary)
                }
                Text(source.excerpt)
                    .font(.caption)
                    .foregroundStyle(MemoTheme.secondary)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
            }
            .padding(13)
            .background(
                RoundedRectangle(cornerRadius: 15)
                    .fill(Color.white.opacity(0.07))
            )
        }
        .buttonStyle(.plain)
    }

    private func sourceTime(_ start: TimeInterval, _ end: TimeInterval?) -> String {
        let startText = MemoFormatters.duration(start)
        guard let end else { return "~\(startText)" }
        return "~\(startText)-\(MemoFormatters.duration(end))"
    }

    private func confirmationCard(_ action: AgentPendingAction) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Confirmation Required", systemImage: "checkmark.shield")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.yellow)
            HStack {
                Button("Cancel", role: .cancel) {
                    Task {
                        await ai.confirmPendingAction(action, confirm: false, store: store)
                    }
                }
                .buttonStyle(.bordered)

                Button(action.type == .deleteRecording ? "Delete" : "Confirm") {
                    Task {
                        await ai.confirmPendingAction(action, confirm: true, store: store)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(action.type == .deleteRecording ? .red : MemoTheme.waveform)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color.yellow.opacity(0.09))
                .stroke(Color.yellow.opacity(0.25))
        )
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Ask your memo agent...", text: $query, axis: .vertical)
                .lineLimit(1...4)
                .padding(.horizontal, 13)
                .padding(.vertical, 11)
                .background(
                    RoundedRectangle(cornerRadius: 18)
                        .fill(Color.white.opacity(0.11))
                )

            Button {
                let value = query
                query = ""
                Task {
                    await ai.ask(
                        value,
                        currentRecordingID: currentRecordingID,
                        store: store
                    )
                }
            } label: {
                Image(systemName: "arrow.up")
                    .font(.system(size: 17, weight: .bold))
                    .frame(width: 42, height: 42)
                    .background(Circle().fill(MemoTheme.waveform))
            }
            .disabled(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}

private struct AISettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var ai: AIController
    @AppStorage(AIConfiguration.serverURLKey) private var serverURL =
        AIConfiguration.defaultServerURL

    var body: some View {
        NavigationStack {
            Form {
                Section("AI Server") {
                    TextField("Server URL", text: $serverURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    Button("Test Connection") {
                        Task { await ai.checkServer() }
                    }
                    if let available = ai.isServerAvailable {
                        Label(
                            available ? "Connected" : "Not Connected",
                            systemImage: available
                                ? "checkmark.circle.fill"
                                : "xmark.circle.fill"
                        )
                        .foregroundStyle(available ? .green : .red)
                    }
                }

                Section("Connection Help") {
                    Text(
                        "Simulator: http://127.0.0.1:8000\n"
                            + "Physical iPhone: use your Mac's .local hostname, "
                            + "for example http://Amy-Mac.local:8000."
                    )
                    .font(.footnote)
                }

                Section("Privacy") {
                    Text(
                        "Recordings stay on the device. Audio is sent to the configured "
                            + "server only when analysis is generated or retried."
                    )
                    .font(.footnote)
                }
            }
            .navigationTitle("AI Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
