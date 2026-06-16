import SwiftUI
import UIKit

@main
struct VoiceMemoAgentApp: App {
    @StateObject private var store = RecordingStore()
    @StateObject private var audio = AudioController()
    @StateObject private var ai = AIController()

    init() {
        configureAppearance()
    }

    var body: some Scene {
        WindowGroup {
            RecordingListView()
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
                .preferredColorScheme(.dark)
        }
    }

    private func configureAppearance() {
        UINavigationBar.appearance().largeTitleTextAttributes = [.foregroundColor: UIColor.white]
        UINavigationBar.appearance().titleTextAttributes = [.foregroundColor: UIColor.white]
    }
}
