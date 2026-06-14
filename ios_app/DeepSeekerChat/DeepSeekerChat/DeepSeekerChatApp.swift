/**
 * Erosolar (DeepSeekerChat) - iOS App
 * Display name "Erosolar", bundle com.erosolarai.chat (see DOMAINS.md + project.yml)
 * Main app entry point
 *
 * Author: Bo Shang <bo@shang.software>
 */

import SwiftUI
import FirebaseCore

@main
struct DeepSeekerChatApp: App {

    init() {
        FirebaseApp.configure()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
