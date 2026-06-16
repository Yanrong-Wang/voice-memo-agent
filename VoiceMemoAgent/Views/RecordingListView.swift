import SwiftUI

struct RecordingListView: View {
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    @State private var selectedID: UUID?
    @State private var searchText = ""
    @State private var isSearching = false
    @State private var showsRecordingSession = false
    @State private var showsPermissionAlert = false
    @State private var showsAgent = false

    private var visibleRecordings: [Recording] {
        guard !searchText.isEmpty else { return store.recordings }
        return store.recordings.filter {
            $0.searchableText.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            MemoTheme.background.ignoresSafeArea()

            VStack(spacing: 0) {
                header
                recordingList
            }

            recordButton
                .padding(.bottom, 10)
        }
        .foregroundStyle(.white)
        .fullScreenCover(isPresented: $showsRecordingSession) {
            RecordingSessionView()
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .sheet(isPresented: $showsAgent) {
            AgentChatView()
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .alert("Microphone Access", isPresented: $showsPermissionAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(audio.lastError ?? "Microphone access is required to record.")
        }
        .alert(
            "Something Went Wrong",
            isPresented: Binding(
                get: { store.lastError != nil },
                set: { if !$0 { store.lastError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { store.lastError = nil }
        } message: {
            Text(store.lastError ?? "")
        }
    }

    private var header: some View {
        VStack(spacing: 12) {
            HStack {
                Button(action: {}) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 19, weight: .semibold))
                        .frame(width: 42, height: 42)
                        .background(Circle().fill(Color.white.opacity(0.1)))
                }
                .accessibilityLabel("Folders")

                Spacer()

                Button {
                    showsAgent = true
                } label: {
                    Image(systemName: "sparkles")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 42, height: 42)
                        .background(Circle().fill(MemoTheme.waveform.opacity(0.2)))
                }
                .accessibilityLabel("Open Memo Agent")

                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isSearching.toggle()
                        if !isSearching { searchText = "" }
                    }
                } label: {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 19, weight: .semibold))
                        .frame(width: 42, height: 42)
                        .background(Circle().fill(Color.white.opacity(0.1)))
                }
                .accessibilityLabel("Search")

                Button("Select") {}
                    .font(.system(size: 17, weight: .medium))
                    .padding(.horizontal, 14)
                    .frame(height: 42)
                    .background(Capsule().fill(Color.white.opacity(0.1)))
            }

            HStack {
                Text("All Recordings")
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                Spacer()
            }

            if isSearching {
                HStack(spacing: 9) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(MemoTheme.secondary)
                    TextField("Search recordings", text: $searchText)
                        .textInputAutocapitalization(.never)
                    if !searchText.isEmpty {
                        Button {
                            searchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(MemoTheme.secondary)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .frame(height: 38)
                .background(RoundedRectangle(cornerRadius: 11).fill(Color.white.opacity(0.12)))
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(MemoTheme.background)
    }

    private var recordingList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                if visibleRecordings.isEmpty {
                    emptyState
                } else {
                    ForEach(visibleRecordings) { recording in
                        RecordingRowView(
                            recording: recording,
                            isExpanded: selectedID == recording.id,
                            onSelect: {
                                withAnimation(.easeInOut(duration: 0.22)) {
                                    if selectedID == recording.id {
                                        selectedID = nil
                                        audio.stopPlayback()
                                    } else {
                                        audio.stopPlayback()
                                        selectedID = recording.id
                                    }
                                }
                            }
                        )
                        .environmentObject(ai)
                    }
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 104)
        }
        .scrollIndicators(.hidden)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: searchText.isEmpty ? "waveform" : "magnifyingglass")
                .font(.system(size: 34, weight: .medium))
                .foregroundStyle(MemoTheme.secondary)
            Text(searchText.isEmpty ? "No Recordings" : "No Results")
                .font(.headline)
            Text(searchText.isEmpty ? "Tap the red button to make your first recording." : "Try a different title, transcript, or tag.")
                .font(.subheadline)
                .foregroundStyle(MemoTheme.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 110)
    }

    private var recordButton: some View {
        Button {
            Task {
                if await audio.startRecording() {
                    showsRecordingSession = true
                } else {
                    showsPermissionAlert = true
                }
            }
        } label: {
            Circle()
                .fill(MemoTheme.record)
                .frame(width: 62, height: 62)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.22), lineWidth: 4)
                        .padding(-5)
                )
                .shadow(color: .black.opacity(0.5), radius: 8, y: 3)
        }
        .accessibilityLabel("Record")
    }
}
