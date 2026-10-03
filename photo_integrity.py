from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "obuladinnesai/claim-photo-integrity-detector-v1"

TILE_SIZE = 224
OVERLAP = 0.25
BATCH_SIZE = 16
SUSPICIOUS_THRESHOLD = 0.80


class PhotoIntegrityDetector:
    """
    Reusable AI-generated / manipulated image detector.

    Example:
        detector = PhotoIntegrityDetector()

        result = detector.analyze("car.jpg")

        print(result["decision"])
        print(result["average_ai_score"])
    """

    def __init__(
        self,
        model_id=MODEL_ID,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        suspicious_threshold=SUSPICIOUS_THRESHOLD,
        device=None,
    ):
        self.model_id = model_id
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.suspicious_threshold = suspicious_threshold

        # Select device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Loading photo integrity model on {self.device}...")

        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id
        )

        # Load model
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_id
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Model labels
        self.labels = self.model.config.id2label

        # Find AI/fake/synthetic label
        self.ai_label = self._find_ai_label()

        print("Photo integrity model loaded.")
        print(f"AI label: {self.ai_label}")

    # ---------------------------------------------------------
    # LABEL DETECTION
    # ---------------------------------------------------------

    def _find_ai_label(self):
        """
        Automatically identify the AI/fake/synthetic class.
        """

        for label in self.labels.values():

            label_lower = label.lower()

            if (
                "ai" in label_lower
                or "fake" in label_lower
                or "synthetic" in label_lower
            ):
                return label

        raise RuntimeError(
            "Could not identify AI/fake label. "
            f"Available labels: {list(self.labels.values())}"
        )

    # ---------------------------------------------------------
    # TILE GENERATION
    # ---------------------------------------------------------

    def generate_tiles(self, image):
        """
        Divide an image into overlapping tiles.

        Returns:
            list of dictionaries containing:
                image
                x
                y
        """

        width, height = image.size

        stride = int(
            self.tile_size * (1 - self.overlap)
        )

        x_positions = list(
            range(
                0,
                max(width - self.tile_size, 0) + 1,
                stride,
            )
        )

        y_positions = list(
            range(
                0,
                max(height - self.tile_size, 0) + 1,
                stride,
            )
        )

        if (
            not x_positions
            or x_positions[-1] + self.tile_size < width
        ):
            x_positions.append(
                max(width - self.tile_size, 0)
            )

        if (
            not y_positions
            or y_positions[-1] + self.tile_size < height
        ):
            y_positions.append(
                max(height - self.tile_size, 0)
            )

        x_positions = sorted(set(x_positions))
        y_positions = sorted(set(y_positions))

        tiles = []

        for y in y_positions:

            for x in x_positions:

                right = min(
                    x + self.tile_size,
                    width
                )

                bottom = min(
                    y + self.tile_size,
                    height
                )

                tile = image.crop(
                    (x, y, right, bottom)
                )

                # Pad small edge tiles
                if tile.size != (
                    self.tile_size,
                    self.tile_size,
                ):

                    padded = Image.new(
                        "RGB",
                        (
                            self.tile_size,
                            self.tile_size,
                        ),
                        (0, 0, 0),
                    )

                    padded.paste(tile, (0, 0))

                    tile = padded

                tiles.append(
                    {
                        "image": tile,
                        "x": x,
                        "y": y,
                    }
                )

        return tiles

    # ---------------------------------------------------------
    # TILE INFERENCE
    # ---------------------------------------------------------

    def _predict_tiles(self, tiles):
        """
        Run model inference on all image tiles.
        """

        all_results = []

        for start in range(
            0,
            len(tiles),
            self.batch_size,
        ):

            batch = tiles[
                start:start + self.batch_size
            ]

            batch_images = [
                item["image"]
                for item in batch
            ]

            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                outputs = self.model(
                    **inputs
                )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            for index, probability in enumerate(
                probabilities
            ):

                tile = batch[index]

                tile_results = {}

                for class_id, score in enumerate(
                    probability
                ):

                    label = self.labels[class_id]

                    tile_results[label] = (
                        score.item()
                    )

                all_results.append(
                    {
                        "x": tile["x"],
                        "y": tile["y"],
                        "results": tile_results,
                    }
                )

        return all_results

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyze(self, image_path):
        """
        Analyze an image for AI/manipulation indicators.

        Args:
            image_path:
                Path or string pointing to image.

        Returns:
            Dictionary containing analysis results.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        original_size = image.size

        # Generate tiles
        tiles = self.generate_tiles(image)

        # Run inference
        all_results = self._predict_tiles(
            tiles
        )

        # Extract AI scores
        tile_scores = []

        for item in all_results:

            score = item["results"].get(
                self.ai_label,
                0.0,
            )

            tile_scores.append(
                {
                    "x": item["x"],
                    "y": item["y"],
                    "score": score,
                }
            )

        # Sort highest first
        tile_scores.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        if not tile_scores:
            raise RuntimeError(
                "No tiles were generated."
            )

        scores = [
            item["score"]
            for item in tile_scores
        ]

        # Average score
        average_ai = (
            sum(scores) / len(scores)
        )

        # Maximum score
        max_ai = max(scores)

        # Top 10%
        top_count = max(
            1,
            int(len(scores) * 0.10),
        )

        top_scores = scores[:top_count]

        top_10_average = (
            sum(top_scores)
            / len(top_scores)
        )

        # Suspicious regions
        suspicious_tiles = [
            item
            for item in tile_scores
            if item["score"]
            >= self.suspicious_threshold
        ]

        # Decision
        if (
            max_ai >= 0.95
            and len(suspicious_tiles) >= 1
        ):

            decision = (
                "Fake Image Detected"
            )

        elif top_10_average >= 0.75:

            decision = (
                "MANIPULATION SUSPECTED"
            )

        elif average_ai >= 0.50:

            decision = "REVIEW"

        else:

            decision = "LIKELY REAL"

        # Return structured result
        return {
            "image": str(image_path),
            "filename": image_path.name,

            "width": original_size[0],
            "height": original_size[1],

            "total_tiles": len(tile_scores),

            "average_ai_score": average_ai,
            "maximum_ai_score": max_ai,
            "top_10_average": top_10_average,

            "suspicious_tile_count": len(
                suspicious_tiles
            ),

            "suspicious_threshold":
                self.suspicious_threshold,

            "decision": decision,

            "ai_label": self.ai_label,

            "top_suspicious_regions":
                tile_scores[:10],

            "suspicious_regions":
                suspicious_tiles,

            "tile_scores":
                tile_scores,
        }

    # ---------------------------------------------------------
    # SIMPLE RESULT
    # ---------------------------------------------------------

    def is_suspicious(self, image_path):
        """
        Simple boolean check.

        Returns:
            True if manipulation is suspected.
            False otherwise.
        """

        result = self.analyze(
            image_path
        )

        return result["decision"] == (
            "MANIPULATION SUSPECTED"
        )