import cv2

from censor_engine._typing import Image


class EdgeDetectionHelpers:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def prepare_mask(self, mask_image: Image) -> Image:
        mask_image = cv2.cvtColor(
            mask_image,
            cv2.COLOR_BGR2GRAY,
        )

        return cv2.GaussianBlur(
            mask_image,
            (3, 3),
            0,
        )  # Minor blur for better results

    def clean_image(self, mask_image: Image) -> Image:
        # Dilute/Erode to Connect Noise
        mask_image = cv2.dilate(mask_image, self.kernel, iterations=2)
        mask_image = cv2.erode(mask_image, self.kernel, iterations=2)
        mask_image = mask_image.astype(np.uint8)
        return cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    def get_auto_tolerances(
        self,
        mask_image: Image,
        multiplier: float,
    ) -> tuple[int, int]:
        mean, stddev = cv2.meanStdDev(mask_image)
        mean_white = mean[0][0]
        stddev_white = stddev[0][0]
        return (
            mean_white - stddev_white * multiplier,
            mean_white + stddev_white * multiplier,
        )

    def process_lines(
        self,
        mask_image: Image,
        tols: tuple[int, int] | None,
        multiplier: float,
    ) -> Image:
        if not tols:
            tols = self.get_auto_tolerances(mask_image, multiplier)

        mask_image[mask_image < tols[0]] = 0
        mask_image[mask_image > tols[1]] = 255
        return cv2.multiply(mask_image, multiplier)  # type: ignore
