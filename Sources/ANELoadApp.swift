import AppKit
import CoreML
import SwiftUI

// Must match make_model.py: 12 x conv3x3, 256 channels, 64x64.
private let gflopPerInference = 58.0
// Load is a duty cycle: in every window the ANE is kept busy for `level` of the time.
private let dutyPeriod: TimeInterval = 0.1
private let workerCount = 2

// MARK: - Engine

/// Runs Core ML predictions on the Neural Engine from background threads.
final class ANELoadEngine: @unchecked Sendable {
    private let models: [MLModel]
    private let input: MLFeatureProvider
    private let lock = NSLock()
    private var level = 0.0
    private var inferences = 0
    private var stopped = false

    init(compiledModel url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        models = try (0..<workerCount).map { _ in try MLModel(contentsOf: url, configuration: config) }

        let x = try MLMultiArray(shape: [1, 256, 64, 64], dataType: .float16)
        x.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            for i in buffer.indices { buffer[i] = Float16(Float(i % 97) / 97) }
        }
        input = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    }

    func start() {
        for model in models {
            let thread = Thread { [self] in run(model) }
            thread.qualityOfService = .userInitiated
            thread.start()
        }
    }

    func stop() { lock.withLock { stopped = true } }

    func setLevel(_ value: Double) { lock.withLock { level = min(max(value, 0), 1) } }

    func takeInferenceCount() -> Int {
        lock.withLock {
            defer { inferences = 0 }
            return inferences
        }
    }

    private func run(_ model: MLModel) {
        let options = MLPredictionOptions()
        var averageDuration = 0.005
        while true {
            let (current, isStopped) = lock.withLock { (level, stopped) }
            if isStopped { return }
            if current <= 0 {
                Thread.sleep(forTimeInterval: 0.05)
                continue
            }
            // Windows are aligned to the clock so all workers idle at the same time.
            let now = Date().timeIntervalSinceReferenceDate
            let windowStart = (now / dutyPeriod).rounded(.down) * dutyPeriod
            let busyEnd = windowStart + dutyPeriod * current
            if current >= 1 || now + averageDuration / 2 < busyEnd {
                let began = Date()
                if (try? model.prediction(from: input, options: options)) != nil {
                    lock.withLock { inferences += 1 }
                }
                averageDuration = 0.9 * averageDuration + 0.1 * Date().timeIntervalSince(began)
            } else {
                Thread.sleep(forTimeInterval: max(windowStart + dutyPeriod - now, 0.001))
            }
        }
    }
}

@available(macOS 14.4, *)
private func neuralEnginePlacement(of url: URL) async -> String {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    do {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        guard case .program(let program) = plan.modelStructure,
              let main = program.functions["main"] else { return "Unknown model structure" }
        var onANE = 0
        var total = 0
        for op in main.block.operations {
            guard let usage = plan.deviceUsage(for: op) else { continue }
            total += 1
            if case .neuralEngine = usage.preferred { onANE += 1 }
        }
        return "\(onANE) of \(total) ops on the Neural Engine"
    } catch {
        return "Compute plan unavailable: \(error.localizedDescription)"
    }
}

private func chipName() -> String {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var buffer = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
    return String(cString: buffer)
}

// MARK: - State

@MainActor
final class LoadController: ObservableObject {
    @Published var percent = 0.0 { didSet { engine?.setLevel(percent / 100) } }
    @Published var status = "Preparing model…"
    @Published var placement = "Checking…"
    @Published var inferencesPerSecond = 0.0
    @Published var ready = false

    private var engine: ANELoadEngine?
    private var timer: Timer?
    private var lastSample = Date()

    func start() async {
        guard engine == nil else { return }
        do {
            guard let source = Bundle.main.url(forResource: "AneStress", withExtension: "mlpackage") else {
                status = "AneStress.mlpackage is missing from the app bundle"
                return
            }
            status = "Compiling model…"
            let compiled = try await MLModel.compileModel(at: source)
            status = "Loading model onto the Neural Engine…"
            let engine = try await Task.detached { try ANELoadEngine(compiledModel: compiled) }.value
            engine.setLevel(percent / 100)
            engine.start()
            self.engine = engine
            ready = true
            status = "Ready"

            lastSample = Date()
            timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.sample() }
            }

            if #available(macOS 14.4, *) {
                placement = await neuralEnginePlacement(of: compiled)
            } else {
                placement = "Needs macOS 14.4 to check"
            }
        } catch {
            status = "Failed: \(error.localizedDescription)"
        }
    }

    func shutdown() {
        timer?.invalidate()
        engine?.stop()
    }

    private func sample() {
        guard let engine else { return }
        let now = Date()
        let elapsed = now.timeIntervalSince(lastSample)
        lastSample = now
        let rate = Double(engine.takeInferenceCount()) / max(elapsed, 0.001)
        inferencesPerSecond = 0.6 * inferencesPerSecond + 0.4 * rate
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var load = LoadController()
    private let chip = chipName()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                Text("Neural Engine load").font(.title2.bold())
                Spacer()
                Text(chip).foregroundStyle(.secondary)
            }

            HStack(spacing: 12) {
                Slider(value: $load.percent, in: 0...100, step: 1)
                Text("\(Int(load.percent))%")
                    .font(.title3.monospacedDigit())
                    .frame(width: 56, alignment: .trailing)
            }
            .disabled(!load.ready)

            HStack {
                ForEach([0, 25, 50, 75, 100], id: \.self) { value in
                    Button("\(value)%") { load.percent = Double(value) }
                }
            }
            .disabled(!load.ready)

            Divider()

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                GridRow {
                    Text("Throughput").foregroundStyle(.secondary)
                    Text(String(format: "%.0f inferences/s", load.inferencesPerSecond)).monospacedDigit()
                }
                GridRow {
                    Text("Compute").foregroundStyle(.secondary)
                    Text(String(format: "≈ %.1f TFLOPS fp16", load.inferencesPerSecond * gflopPerInference / 1000))
                        .monospacedDigit()
                }
                GridRow {
                    Text("Placement").foregroundStyle(.secondary)
                    Text(load.placement)
                }
                GridRow {
                    Text("Status").foregroundStyle(.secondary)
                    Text(load.status)
                }
            }

            Text("The level is a duty cycle: in every 100 ms window the Neural Engine runs for that share of the time and idles for the rest. Check it with: sudo powermetrics --samplers cpu_power -i 1000 | grep -i ane")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .padding(20)
        .frame(width: 460)
        .task { await load.start() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            load.shutdown()
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct ANELoadApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("ANE Load") { ContentView() }
            .windowResizability(.contentSize)
    }
}
