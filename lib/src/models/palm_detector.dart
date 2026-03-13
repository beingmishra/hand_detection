import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:meta/meta.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert/flutter_litert.dart' as litert
    show normalizeRadians, sigmoid, weightedNms;
import '../util/image_utils.dart';

/// A detected palm with rotation rectangle parameters.
///
/// Used to crop and rotate hand regions for landmark extraction.
class PalmDetection {
  /// Size of the square rotation rectangle (normalized).
  final double sqnRrSize;

  /// Rotation angle in radians.
  final double rotation;

  /// Center X coordinate (normalized 0-1).
  final double sqnRrCenterX;

  /// Center Y coordinate (normalized 0-1).
  final double sqnRrCenterY;

  /// Detection confidence score (0.0 to 1.0).
  final double score;

  /// Creates a palm detection result.
  const PalmDetection({
    required this.sqnRrSize,
    required this.rotation,
    required this.sqnRrCenterX,
    required this.sqnRrCenterY,
    required this.score,
  });
}

/// SSD-based palm detector for Stage 1 of the hand detection pipeline.
///
/// Detects palm locations in images using a Single Shot Detector (SSD) architecture
/// with anchor-based decoding. Returns rotation rectangles suitable for cropping
/// hand regions for landmark extraction.
///
/// This is a direct port of the Python PalmDetection class.
class PalmDetector {
  final InterpreterPool _pool = InterpreterPool(poolSize: 1);
  bool _isInitialized = false;

  /// Input dimensions (192x192 for palm detection model).
  late int _inH;
  late int _inW;

  /// Pre-generated SSD anchors.
  late List<List<double>> _anchors;

  /// Preprocessing state - matches Python's calculation.
  /// These use original image dimensions like Python does.
  int _imageHeight = 0;
  int _imageWidth = 0;

  /// square_standard_size = max(image_height, image_width)
  int _squareStandardSize = 0;

  /// square_padding_half_size = abs(image_height - image_width) // 2
  int _squarePaddingHalfSize = 0;

  /// Score threshold for detection filtering.
  final double scoreThreshold;

  /// Pre-allocated input buffer.
  Float32List? _inputBuffer;

  /// Pre-allocated box regressor output [1, 2016, 18].
  List<List<List<double>>>? _outputBoxes;

  /// Pre-allocated classification score output [1, 2016, 1].
  List<List<List<double>>>? _outputScores;

  /// Creates a palm detector with the specified score threshold.
  PalmDetector({this.scoreThreshold = 0.45});

  /// Initializes the palm detector by loading the TFLite model.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/hand_detection/assets/models/hand_detection.tflite';
    await _initWith(
      (options) async => Interpreter.fromAsset(assetPath, options: options),
      performanceConfig,
    );
  }

  /// Initializes the palm detector from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
  }) async {
    await _initWith(
      (options) async => Interpreter.fromBuffer(modelBytes, options: options),
      performanceConfig,
    );
  }

  Future<void> _initWith(
    Future<Interpreter> Function(InterpreterOptions) loader,
    PerformanceConfig? performanceConfig,
  ) async {
    if (_isInitialized) await dispose();
    await _pool.initialize(
      (options, _) async {
        final interpreter = await loader(options);
        interpreter.allocateTensors();
        _setupAnchorsAndBuffers(interpreter);
        return interpreter;
      },
      performanceConfig: performanceConfig,
    );
    _isInitialized = true;
  }

  void _setupAnchorsAndBuffers(Interpreter interpreter) {
    final inTensor = interpreter.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    final anchorOptions = SSDAnchorOptions(
      numLayers: 4,
      minScale: 0.1484375,
      maxScale: 0.75,
      inputSizeHeight: _inH,
      inputSizeWidth: _inW,
      anchorOffsetX: 0.5,
      anchorOffsetY: 0.5,
      strides: [8, 16, 16, 16],
      aspectRatios: [1.0],
      reduceBoxesInLowestLayer: false,
      interpolatedScaleAspectRatio: 1.0,
      fixedAnchorSize: true,
    );
    _anchors = generateAnchors(anchorOptions);

    final numAnchors = _anchors.length;
    _outputBoxes = List.generate(
      1,
      (_) => List.generate(
        numAnchors,
        (_) => List<double>.filled(18, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );
    _outputScores = List.generate(
      1,
      (_) => List.generate(
        numAnchors,
        (_) => List<double>.filled(1, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );
  }

  /// Returns true if the detector has been initialized.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases resources.
  Future<void> dispose() async {
    await _pool.dispose();
    _inputBuffer = null;
    _outputBoxes = null;
    _outputScores = null;
    _isInitialized = false;
  }

  /// Detects palms in the given image.
  ///
  /// Returns a list of [PalmDetection] objects containing rotation rectangle
  /// parameters for each detected palm.
  Future<List<PalmDetection>> detectOnMat(cv.Mat image) async {
    if (!_isInitialized) {
      throw StateError('PalmDetector not initialized.');
    }

    _imageHeight = image.rows;
    _imageWidth = image.cols;

    _squareStandardSize = math.max(_imageHeight, _imageWidth);
    _squarePaddingHalfSize = (_imageHeight - _imageWidth).abs() ~/ 2;

    final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
      image,
      _inW,
      _inH,
    );

    final inputSize = _inH * _inW * 3;
    _inputBuffer ??= Float32List(inputSize);
    ImageUtils.matToFloat32Tensor(paddedImage, buffer: _inputBuffer);

    resizedImage.dispose();
    paddedImage.dispose();

    final inputs = [_inputBuffer!.buffer];
    final outputs = <int, Object>{
      0: _outputBoxes!,
      1: _outputScores!,
    };

    await _pool.withInterpreter((interp, iso) async {
      if (iso != null) {
        await iso.runForMultipleInputs(inputs, outputs);
      } else {
        interp.runForMultipleInputs(inputs, outputs);
      }
    });

    final decodedBoxes = _decodeBoxes(
      _outputBoxes![0],
      _outputScores![0],
    );

    return _postprocess(decodedBoxes);
  }

  /// Decodes raw box predictions using anchors.
  ///
  /// Returns decoded boxes as [score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y].
  ///
  /// Optimization: Only decodes the 4 coordinate pairs actually used (indices 0-5, 8-9)
  /// instead of all 9 pairs (18 values), avoiding intermediate list allocation.
  /// Raw box layout: [cx, cy, w, h, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, ...] where each value is offset relative to anchor, scaled by 192.
  List<List<double>> _decodeBoxes(
    List<List<double>> rawBoxes,
    List<List<double>> rawScores, {
    double scale = 192.0,
  }) {
    final results = <List<double>>[];
    final invScale = 1.0 / scale;

    for (int i = 0; i < rawBoxes.length; i++) {
      final rawScore = rawScores[i][0];
      final score = litert.sigmoid(rawScore);

      if (score <= scoreThreshold) continue;

      final rawBox = rawBoxes[i];
      final anchor = _anchors[i];

      final anchorW = anchor[2];
      final anchorH = anchor[3];
      final anchorX = anchor[0];
      final anchorY = anchor[1];

      final cx = rawBox[0] * anchorW * invScale + anchorX;
      final cy = rawBox[1] * anchorH * invScale + anchorY;

      final wPoint = rawBox[2] * anchorW * invScale + anchorX;
      final hPoint = rawBox[3] * anchorH * invScale + anchorY;
      final w = wPoint - anchorX;
      final h = hPoint - anchorY;
      final boxSize = math.max(w, h);

      final kp0X = rawBox[4] * anchorW * invScale + anchorX;
      final kp0Y = rawBox[5] * anchorH * invScale + anchorY;

      final kp2X = rawBox[8] * anchorW * invScale + anchorX;
      final kp2Y = rawBox[9] * anchorH * invScale + anchorY;

      results.add([score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y]);
    }

    return results;
  }

  /// Post-processes decoded boxes to produce palm detections.
  ///
  /// Transforms coordinates from model space back to original image space,
  /// accounting for the padding applied during preprocessing.
  /// Matches Python's __postprocess implementation.
  List<PalmDetection> _postprocess(List<List<double>> boxes) {
    if (boxes.isEmpty) return [];

    final palms = <PalmDetection>[];

    for (final box in boxes) {
      final pdScore = box[0];
      final boxX = box[1];
      final boxY = box[2];
      final boxSize = box[3];
      final kp0X = box[4];
      final kp0Y = box[5];
      final kp2X = box[6];
      final kp2Y = box[7];

      if (boxSize > 0) {
        final kp02X = kp2X - kp0X;
        final kp02Y = kp2Y - kp0Y;
        var sqnRrSize = 2.9 * boxSize;
        var rotation = 0.5 * math.pi - math.atan2(-kp02Y, kp02X);
        rotation = litert.normalizeRadians(rotation);
        var sqnRrCenterX = boxX + 0.5 * boxSize * math.sin(rotation);
        var sqnRrCenterY = boxY - 0.5 * boxSize * math.cos(rotation);

        if (_imageHeight > _imageWidth) {
          sqnRrCenterX =
              (sqnRrCenterX * _squareStandardSize - _squarePaddingHalfSize) /
                  _imageWidth;
        } else {
          sqnRrCenterY =
              (sqnRrCenterY * _squareStandardSize - _squarePaddingHalfSize) /
                  _imageHeight;
        }

        palms.add(PalmDetection(
          sqnRrSize: sqnRrSize,
          rotation: rotation,
          sqnRrCenterX: sqnRrCenterX,
          sqnRrCenterY: sqnRrCenterY,
          score: pdScore,
        ));
      }
    }

    return _nms(palms);
  }

  /// Weighted Non-Maximum Suppression for palm detections.
  ///
  /// Fuses overlapping boxes by score-weighted coordinate averaging,
  /// producing tighter bounding boxes from the many overlapping SSD anchors
  /// that fire on the same palm. Keeps the highest-scoring detection's
  /// rotation (derived from keypoints) while averaging center and size.
  List<PalmDetection> _nms(List<PalmDetection> palms,
      {double iouThreshold = 0.45}) {
    if (palms.isEmpty) return palms;
    final sorted = List<PalmDetection>.from(palms)
      ..sort((a, b) => b.score.compareTo(a.score));
    final boxes = sorted.map(_palmToXYXY).toList();
    final scores = sorted.map((p) => p.score).toList();
    final results = litert.weightedNms(boxes, scores, iouThres: iouThreshold);
    final maxDim = math.max(_imageWidth, _imageHeight).toDouble();
    return [
      for (final r in results)
        PalmDetection(
          sqnRrSize:
              math.max(r.box[2] - r.box[0], r.box[3] - r.box[1]) / maxDim,
          rotation: sorted[r.index].rotation,
          sqnRrCenterX: (r.box[0] + r.box[2]) / 2 / _imageWidth,
          sqnRrCenterY: (r.box[1] + r.box[3]) / 2 / _imageHeight,
          score: r.score,
        ),
    ];
  }

  /// Converts a PalmDetection to an XYXY bounding box [left, top, right, bottom].
  List<double> _palmToXYXY(PalmDetection p) {
    final maxDim = math.max(_imageWidth, _imageHeight).toDouble();
    final halfSize = (p.sqnRrSize * maxDim) / 2;
    final centerX = p.sqnRrCenterX * _imageWidth;
    final centerY = p.sqnRrCenterY * _imageHeight;
    return [
      centerX - halfSize,
      centerY - halfSize,
      centerX + halfSize,
      centerY + halfSize,
    ];
  }

  /// Exposes anchor generation for testing.
  @visibleForTesting
  List<List<double>> get anchorsForTest => _anchors;

  /// Exposes input width for testing.
  @visibleForTesting
  int get inputWidth => _inW;

  /// Exposes input height for testing.
  @visibleForTesting
  int get inputHeight => _inH;
}
