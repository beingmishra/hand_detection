import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../models/palm_detector.dart';

/// Utility functions for image preprocessing and transformations using OpenCV.
///
/// Provides letterbox preprocessing, coordinate transformations, tensor
/// conversion utilities, and rotation-aware cropping for hand detection.
/// Uses native OpenCV operations for 10-50x better performance than pure Dart.
class ImageUtils {
  /// Keeps aspect ratio while resizing and centers with padding.
  ///
  /// This matches the Python keep_aspect_resize_and_pad function.
  /// Uses OpenCV's native resize for significantly better performance.
  static (cv.Mat padded, cv.Mat resized) keepAspectResizeAndPad(
    cv.Mat image,
    int resizeWidth,
    int resizeHeight,
  ) {
    final params = computeAspectPadParams(
      srcWidth: image.cols,
      srcHeight: image.rows,
      targetWidth: resizeWidth,
      targetHeight: resizeHeight,
    );

    final resizedImage = cv.resize(
      image,
      (params.newWidth, params.newHeight),
      interpolation: cv.INTER_LINEAR,
    );

    final paddedImage = cv.copyMakeBorder(
      resizedImage,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );

    return (paddedImage, resizedImage);
  }

  /// Crops a rotated rectangle from an image using OpenCV's warpAffine.
  ///
  /// This is used to extract hand regions with proper rotation alignment
  /// for the landmark model. Uses OpenCV's SIMD-optimized warpAffine which
  /// is 10-50x faster than pure Dart bilinear interpolation.
  ///
  /// Parameters:
  /// - [image]: Source image
  /// - [palm]: Palm detection containing rotation rectangle parameters
  /// - [padding]: Whether to handle edge cases (always handled via border mode)
  ///
  /// Returns the cropped and rotated hand image, or null if the crop is invalid.
  static cv.Mat? rotateAndCropRectangle(
    cv.Mat image,
    PalmDetection palm, {
    bool padding = true,
  }) {
    final imageWidth = image.cols;
    final imageHeight = image.rows;

    final cx = palm.sqnRrCenterX * imageWidth;
    final cy = palm.sqnRrCenterY * imageHeight;

    final size = (palm.sqnRrSize * math.max(imageWidth, imageHeight)).round();
    if (size <= 0) return null;

    final angleDegrees = palm.rotation * 180.0 / math.pi;

    final rotMat =
        cv.getRotationMatrix2D(cv.Point2f(cx, cy), angleDegrees, 1.0);

    final outCx = size / 2.0;
    final outCy = size / 2.0;

    final tx = rotMat.at<double>(0, 2) + outCx - cx;
    final ty = rotMat.at<double>(1, 2) + outCy - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    final output = cv.warpAffine(
      image,
      rotMat,
      (size, size),
      borderMode: cv.BORDER_CONSTANT,
      borderValue: cv.Scalar.black,
    );

    rotMat.dispose();
    return output;
  }

  /// Creates a rotated rectangle crop info from palm detection.
  ///
  /// Returns the crop parameters needed for landmark extraction:
  /// [cx, cy, width, height, angleDegrees]
  static List<double> palmToRect(
      PalmDetection palm, int imageWidth, int imageHeight) {
    final cx = palm.sqnRrCenterX * imageWidth;
    final cy = palm.sqnRrCenterY * imageHeight;
    final size = palm.sqnRrSize * math.max(imageWidth, imageHeight);
    final angleDegrees = palm.rotation * 180.0 / math.pi;

    return [cx, cy, size, size, angleDegrees];
  }

  /// Applies letterbox preprocessing to fit an image into target dimensions.
  ///
  /// Scales the source image to fit within [tw]x[th] while maintaining aspect ratio,
  /// then pads with gray (114, 114, 114) to fill the target dimensions.
  ///
  /// This is critical for YOLO-style object detection models that expect fixed input sizes.
  ///
  /// Parameters:
  /// - [src]: Source image to preprocess
  /// - [tw]: Target width in pixels
  /// - [th]: Target height in pixels
  /// - [ratioOut]: Output parameter that receives the scale ratio used
  /// - [dwdhOut]: Output parameter that receives padding [dw, dh] values
  ///
  /// Returns the letterboxed image with dimensions [tw]x[th].
  static cv.Mat letterbox(
    cv.Mat src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    final params = computeLetterboxParams(
      srcWidth: src.cols,
      srcHeight: src.rows,
      targetWidth: tw,
      targetHeight: th,
    );

    final resized = cv.resize(
      src,
      (params.newWidth, params.newHeight),
      interpolation: cv.INTER_LINEAR,
    );

    final canvas = cv.copyMakeBorder(
      resized,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    resized.dispose();

    ratioOut
      ..clear()
      ..add(params.scale);
    dwdhOut
      ..clear()
      ..addAll([params.padLeft, params.padTop]);
    return canvas;
  }

  /// Applies letterbox preprocessing to 256x256 dimensions.
  ///
  /// Convenience method that calls [letterbox] with fixed 256x256 target size.
  /// Used for hand landmark model preprocessing.
  static cv.Mat letterbox256(
    cv.Mat src,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    return letterbox(src, 256, 256, ratioOut, dwdhOut);
  }

  /// Applies letterbox preprocessing to 224x224 dimensions.
  ///
  /// Convenience method that calls [letterbox] with fixed 224x224 target size.
  /// Used for hand landmark model preprocessing (MediaPipe format).
  static cv.Mat letterbox224(
    cv.Mat src,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    return letterbox(src, 224, 224, ratioOut, dwdhOut);
  }

  /// Converts a cv.Mat to a flat Float32List tensor for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns a flat Float32List with normalized RGB pixel values.
  static Float32List matToFloat32Tensor(cv.Mat mat, {Float32List? buffer}) {
    return bgrBytesToRgbFloat32(
      bytes: mat.data,
      totalPixels: mat.rows * mat.cols,
      buffer: buffer,
    );
  }

  /// Converts an image to a 4D tensor in NHWC format for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  /// The output format is [batch, height, width, channels] where batch=1 and channels=3 (RGB).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [width]: Target width (must match mat.cols)
  /// - [height]: Target height (must match mat.rows)
  /// - [reuse]: Optional tensor buffer to reuse (must match dimensions)
  ///
  /// Returns a 4D tensor [1, height, width, 3] with normalized pixel values.
  static List<List<List<List<double>>>> matToNHWC4D(
    cv.Mat mat,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final out = reuse ?? createNHWCTensor4D(height, width);
    fillNHWC4DFromBgrBytes(
        bytes: mat.data, tensor: out, width: width, height: height);
    return out;
  }
}
