# import argparse
# import json
# import sys
# from pathlib import Path
# import contextlib
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval


# ZERO_METRICS = {
#     "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
#     "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
#     "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
#     "prediction_count": 0,
# }


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ground-truth", required=True, type=Path)
#     parser.add_argument("--predictions", required=True, type=Path)
#     parser.add_argument("--task", choices=["bbox", "segm"], required=True)
#     args = parser.parse_args()

#     preds = json.loads(args.predictions.read_text(encoding="utf-8"))
#     gt_data = json.loads(args.ground_truth.read_text(encoding="utf-8"))

#     if not preds:
#         metrics = ZERO_METRICS.copy()
#         print(json.dumps(metrics, indent=2))
#         return

#     allowed_image_ids = {int(x["id"]) for x in gt_data["images"]}
#     allowed_category_ids = {int(c["id"]) for c in gt_data["categories"]}

#     normalized = []
#     for pred in preds:
#         p = dict(pred)
#         if args.task == "segm":
#             p.pop("bbox", None)
#         normalized.append(p)
#     try:
#         # Keep COCO's progress messages out of the JSON output.
#         with contextlib.redirect_stdout(sys.stderr):

#             gt = COCO(str(args.ground_truth))

#             gt.dataset.setdefault("info", {})
#             gt.dataset.setdefault("licenses", [])

#             dt = gt.loadRes(normalized)

#             evaluator = COCOeval(
#                 gt,
#                 dt,
#                 iouType=args.task
#             )

#             evaluator.params.imgIds = sorted(allowed_image_ids)
#             evaluator.params.catIds = sorted(allowed_category_ids)

#             # Actual evaluation
#             evaluator.evaluate()
#             evaluator.accumulate()

#             # IMPORTANT: Generates evaluator.stats
#             evaluator.summarize()

#         s = evaluator.stats

#         if len(s) < 9:
#             raise RuntimeError(
#                 "COCOeval failed to generate all 9 evaluation metrics."
#             )

#         metrics = {
#             "AP": float(s[0]),
#             "AP50": float(s[1]),
#             "AP75": float(s[2]),
#             "AP_small": float(s[3]),
#             "AP_medium": float(s[4]),
#             "AP_large": float(s[5]),
#             "AR1": float(s[6]),
#             "AR10": float(s[7]),
#             "AR100": float(s[8]),
#             "prediction_count": len(preds)
#         }

#     except Exception as e:
#         raise RuntimeError(
#             f"Competition evaluation failed: {e}"
#         ) from e

#     print(json.dumps(metrics, indent=2))


# if __name__ == "__main__":
#     main()



import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


ZERO_METRICS = {
    "AP": 0.0,
    "AP50": 0.0,
    "AP75": 0.0,
    "AP_small": 0.0,
    "AP_medium": 0.0,
    "AP_large": 0.0,
    "AR1": 0.0,
    "AR10": 0.0,
    "AR100": 0.0,
    "prediction_count": 0,
}


def segmentation_to_rle(segmentation, height, width, index):
    """
    Convert COCO polygon or uncompressed RLE segmentation
    into compressed RLE required by COCO.loadRes().
    """

    # CASE 1: COCO polygon segmentation
    if isinstance(segmentation, list):

        if not segmentation:
            raise ValueError(
                f"Prediction {index}: empty polygon segmentation."
            )

        polygons = []

        for polygon_index, polygon in enumerate(segmentation):

            if not isinstance(polygon, list):
                raise ValueError(
                    f"Prediction {index}: polygon {polygon_index} "
                    "must be a list."
                )

            if len(polygon) < 6 or len(polygon) % 2 != 0:
                raise ValueError(
                    f"Prediction {index}: polygon {polygon_index} "
                    "must contain at least 3 coordinate pairs."
                )

            coordinates = [float(v) for v in polygon]

            if not all(math.isfinite(v) for v in coordinates):
                raise ValueError(
                    f"Prediction {index}: polygon contains "
                    "invalid coordinates."
                )

            xs = coordinates[0::2]
            ys = coordinates[1::2]

            if (
                min(xs) < 0 or max(xs) > width
                or min(ys) < 0 or max(ys) > height
            ):
                raise ValueError(
                    f"Prediction {index}: polygon coordinates "
                    "exceed image boundaries."
                )

            polygons.append(coordinates)

        # Convert polygons to RLE.
        rles = maskUtils.frPyObjects(
            polygons,
            height,
            width
        )

        if not rles:
            raise ValueError(
                f"Prediction {index}: polygon conversion failed."
            )

        # Merge multiple polygons into one instance mask.
        return maskUtils.merge(rles)

    # CASE 2: COCO RLE segmentation
    if isinstance(segmentation, dict):

        size = segmentation.get("size")
        counts = segmentation.get("counts")

        if (
            not isinstance(size, list)
            or len(size) != 2
            or [int(v) for v in size] != [height, width]
        ):
            raise ValueError(
                f"Prediction {index}: RLE dimensions do not "
                "match the corresponding image."
            )

        # Uncompressed RLE -> compressed RLE
        if isinstance(counts, list):

            rle = maskUtils.frPyObjects(
                segmentation,
                height,
                width
            )

            if isinstance(rle, list):
                rle = maskUtils.merge(rle)

            return rle

        # JSON compressed RLE stores counts as a string.
        if isinstance(counts, str) and counts:

            return {
                "size": [height, width],
                "counts": counts.encode("ascii")
            }

        # Already compressed in memory.
        if isinstance(counts, bytes) and counts:

            return segmentation

        raise ValueError(
            f"Prediction {index}: invalid RLE counts."
        )

    raise ValueError(
        f"Prediction {index}: segmentation must be "
        "a polygon list or COCO RLE dictionary."
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--task",
        choices=["bbox", "segm"],
        required=True
    )

    args = parser.parse_args()

    preds = json.loads(
        args.predictions.read_text(encoding="utf-8")
    )

    gt_data = json.loads(
        args.ground_truth.read_text(encoding="utf-8")
    )

    if not isinstance(preds, list):
        raise ValueError(
            "Submission must be a JSON array."
        )

    if not preds:
        print(json.dumps(ZERO_METRICS, indent=2))
        return

    image_by_id = {
        int(image["id"]): image
        for image in gt_data["images"]
    }

    allowed_image_ids = set(image_by_id.keys())

    allowed_category_ids = {
        int(category["id"])
        for category in gt_data["categories"]
    }

    normalized = []

    for index, prediction in enumerate(preds):

        if not isinstance(prediction, dict):
            raise ValueError(
                f"Prediction {index} must be a JSON object."
            )

        p = dict(prediction)

        try:
            image_id = int(p["image_id"])
            category_id = int(p["category_id"])
            score = float(p["score"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Prediction {index}: invalid image_id, "
                "category_id or score."
            )

        if image_id not in allowed_image_ids:
            raise ValueError(
                f"Prediction {index}: unknown image ID {image_id}."
            )

        if category_id not in allowed_category_ids:
            raise ValueError(
                f"Prediction {index}: unknown category ID "
                f"{category_id}."
            )

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"Prediction {index}: invalid confidence score."
            )

        p["image_id"] = image_id
        p["category_id"] = category_id
        p["score"] = score

        if args.task == "segm":

            image = image_by_id[image_id]

            height = int(image["height"])
            width = int(image["width"])

            if "segmentation" not in p:
                raise ValueError(
                    f"Prediction {index}: missing segmentation."
                )

            # Force COCO to use its segmentation branch.
            p.pop("bbox", None)

            # IMPORTANT: Polygon -> compressed RLE
            p["segmentation"] = segmentation_to_rle(
                p["segmentation"],
                height,
                width,
                index
            )

        else:

            if "bbox" not in p:
                raise ValueError(
                    f"Prediction {index}: missing bbox."
                )

        normalized.append(p)

    try:

        # Keep COCO logs away from metrics.json.
        with contextlib.redirect_stdout(sys.stderr):

            gt = COCO(str(args.ground_truth))

            gt.dataset.setdefault("info", {})
            gt.dataset.setdefault("licenses", [])

            dt = gt.loadRes(normalized)

            evaluator = COCOeval(
                gt,
                dt,
                iouType=args.task
            )

            evaluator.params.imgIds = sorted(
                allowed_image_ids
            )

            evaluator.params.catIds = sorted(
                allowed_category_ids
            )

            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        stats = evaluator.stats

        if len(stats) < 9:
            raise RuntimeError(
                "COCO evaluation did not generate all metrics."
            )

        metrics = {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR1": float(stats[6]),
            "AR10": float(stats[7]),
            "AR100": float(stats[8]),
            "prediction_count": len(preds)
        }

    except Exception as error:

        raise RuntimeError(
            f"Competition evaluation failed: {error}"
        ) from error

    # Only valid JSON goes to stdout.
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
