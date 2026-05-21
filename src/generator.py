import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class DottedLabelGenerator:
    def __init__(self, output_dir="data/synthetic"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Characters to use in synthetic data
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        
    def create_dot_matrix_text(self, text, font_size=40):
        """
        Improved dot matrix simulation: creates characters from a grid of dots.
        """
        # Character map for a 5x7 dot matrix (simplified example for 0-9 and A-Z)
        # In a real scenario, we'd use a .ttf dot-matrix font, but we'll simulate it.
        img_w = len(text) * (font_size // 2 + 10)
        img_h = font_size + 20
        dot_image = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # We'll use PIL to get a high-res mask then aggressively downsample
        canvas = Image.new('L', (img_w * 2, img_h * 2), 0)
        draw = ImageDraw.Draw(canvas)
        try:
            # Try to use a thicker font for better dot coverage
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((10, 10), text, font=font, fill=255)
        mask = np.array(canvas)
        
        # Grid spacing for 'inkjet' dots
        dot_spacing = random.randint(6, 10)
        dot_radius = random.randint(1, 2)
        
        for y in range(0, mask.shape[0], dot_spacing):
            for x in range(0, mask.shape[1], dot_spacing):
                if mask[y, x] > 100:
                    # Scale back to original size
                    cv2.circle(dot_image, (x//2, y//2), dot_radius, 255, -1)
                    
        return dot_image

    def apply_distortions(self, image):
        """
        Applies metallic texture, curvature, and noise.
        """
        h, w = image.shape
        # Create metallic background
        bg = np.random.normal(180, 20, (h, w)).astype(np.uint8)
        
        # Add some 'brushed metal' lines
        for _ in range(10):
            y_pos = random.randint(0, h)
            cv2.line(bg, (0, y_pos), (w, y_pos + random.randint(-5, 5)), (160, 160, 160), 1)

        # Merge text with background (black ink on metal)
        # text is 255 where dots are, 0 otherwise
        final = bg.copy()
        mask = image > 127
        # Randomize ink darkness
        ink_color = random.randint(30, 70)
        final[mask] = ink_color
        
        # Curvature (Cylindrical warp)
        # This is simplified: we shift pixels vertically based on a parabola
        curve_intensity = random.uniform(-0.001, 0.001)
        center_x = w // 2
        for x in range(w):
            offset = int(curve_intensity * (x - center_x)**2)
            if offset != 0:
                M = np.float32([[1, 0, 0], [0, 1, offset]])
                final[:, x] = cv2.warpAffine(final[:, x:x+1], M, (1, h))[:, 0]

        # Add glare (Gaussian light spot)
        glare = np.zeros((h, w), dtype=np.uint8)
        cx, cy = random.randint(0, w), random.randint(0, h)
        strength = random.randint(50, 150)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i-cy)**2 + (j-cx)**2)
                glare[i, j] = max(0, strength - dist * 0.5)
        
        final = cv2.addWeighted(final, 1.0, glare, 0.5, 0)
        
        return final

    def generate_batch(self, count=100):
        for i in range(count):
            # Generate a random expiry date like string
            day = random.randint(1, 28)
            month = random.randint(1, 12)
            year = random.randint(24, 30)
            text = f"EXP {day:02d}/{month:02d}/{year:02d}"
            
            img = self.create_dot_matrix_text(text)
            img = self.apply_distortions(img)
            
            filename = f"dotted_{i:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, img)
            
            # Save label for training
            with open(os.path.join(self.output_dir, filename.replace(".png", ".txt")), "w") as f:
                f.write(text)

if __name__ == "__main__":
    gen = DottedLabelGenerator()
    print("Generating 1000 synthetic samples...")
    gen.generate_batch(1000)
    print("Done.")
