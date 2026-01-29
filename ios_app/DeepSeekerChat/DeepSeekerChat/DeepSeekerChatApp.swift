/**
 * DeepSeeker Chat - iOS App
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
