/**
 * DeepSeeker Chat - Chat Service Tests
 * Benchmarks and tests for local caching and API connectivity
 *
 * Author: Bo Shang <bo@shang.software>
 */

import XCTest
import FirebaseCore
import FirebaseFirestore
@testable import DeepSeekerChat

final class ChatServiceTests: XCTestCase {

    var chatService: ChatService!

    override func setUp() async throws {
        // Initialize Firebase if not already done
        if FirebaseApp.app() == nil {
            FirebaseApp.configure()
        }
        chatService = await ChatService()
    }

    override func tearDown() async throws {
        chatService = nil
    }

    // MARK: - API Connectivity Tests

    func testAPIEndpointReachable() async throws {
        let url = URL(string: "https://erosolar-api-13762901352.us-central1.run.app/health")!
        let (_, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            XCTFail("Invalid response type")
            return
        }

        XCTAssertEqual(httpResponse.statusCode, 200, "API should be reachable")
    }

    func testAPIResponseFormat() async throws {
        let url = URL(string: "https://erosolar-api-13762901352.us-central1.run.app/v1/responses")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "model": "erosolar",
            "input": [["role": "user", "content": [["type": "input_text", "text": "Hi"]]]],
            "stream": false,
            "max_output_tokens": 50
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            XCTFail("Invalid response type")
            return
        }

        XCTAssertEqual(httpResponse.statusCode, 200, "API should return 200")
        XCTAssertFalse(data.isEmpty, "Response should have content")
    }

    // MARK: - Firebase Offline Cache Tests

    func testFirestoreOfflinePersistence() async throws {
        let db = Firestore.firestore()

        // Verify offline persistence is enabled
        let settings = db.settings
        XCTAssertTrue(settings.cacheSettings is PersistentCacheSettings, "Firestore should have persistent cache enabled")
    }

    func testCacheWriteAndRead() async throws {
        let db = Firestore.firestore()
        let testId = UUID().uuidString
        let testMessage = [
            "id": testId,
            "role": "assistant",
            "content": "Test cached response",
            "timestamp": Timestamp(date: Date())
        ] as [String: Any]

        // Write to cache
        try await db.collection("ios_cache_test")
            .document(testId)
            .setData(testMessage)

        // Read back
        let snapshot = try await db.collection("ios_cache_test")
            .document(testId)
            .getDocument()

        XCTAssertTrue(snapshot.exists, "Cached document should exist")
        XCTAssertEqual(snapshot.data()?["content"] as? String, "Test cached response")

        // Cleanup
        try await db.collection("ios_cache_test").document(testId).delete()
    }

    // MARK: - Performance Benchmarks

    func testAPILatencyBenchmark() async throws {
        let url = URL(string: "https://erosolar-api-13762901352.us-central1.run.app/v1/responses")!

        measure {
            let expectation = self.expectation(description: "API call")

            Task {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                let body: [String: Any] = [
                    "model": "erosolar",
                    "input": [["role": "user", "content": [["type": "input_text", "text": "1+1"]]]],
                    "stream": false,
                    "max_output_tokens": 10
                ]
                request.httpBody = try? JSONSerialization.data(withJSONObject: body)

                do {
                    let _ = try await URLSession.shared.data(for: request)
                } catch {
                    // Ignore errors in benchmark
                }
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 30.0)
        }
    }

    func testFirestoreCacheLatencyBenchmark() async throws {
        let db = Firestore.firestore()
        let testId = UUID().uuidString

        // Pre-write data
        try await db.collection("ios_cache_benchmark")
            .document(testId)
            .setData(["content": "benchmark data", "timestamp": Timestamp(date: Date())])

        measure {
            let expectation = self.expectation(description: "Cache read")

            Task {
                let _ = try? await db.collection("ios_cache_benchmark")
                    .document(testId)
                    .getDocument(source: .cache)
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 5.0)
        }

        // Cleanup
        try await db.collection("ios_cache_benchmark").document(testId).delete()
    }

    // MARK: - Message Flow Tests

    @MainActor
    func testMessageCreation() async throws {
        let message = ChatMessage(role: .user, content: "Hello")

        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, "Hello")
        XCTAssertFalse(message.isStreaming)
        XCTAssertNotNil(message.id)
    }

    @MainActor
    func testChatServiceInitialization() async throws {
        XCTAssertNotNil(chatService)
        XCTAssertTrue(chatService.messages.isEmpty)
        XCTAssertFalse(chatService.isStreaming)
    }

    // MARK: - Error Handling Tests

    func testErrorDetailsClipboardFormat() {
        let details = ErrorDetails(
            timestamp: Date(),
            endpoint: "https://test.com/api",
            statusCode: 500,
            errorMessage: "Test error",
            requestBody: "test request",
            responseBody: "test response"
        )

        let text = details.clipboardText
        XCTAssertTrue(text.contains("DeepSeeker Error Report"))
        XCTAssertTrue(text.contains("500"))
        XCTAssertTrue(text.contains("Test error"))
    }
}
