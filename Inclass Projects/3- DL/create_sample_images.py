import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Optional PyTorch imports - will fallback to synthetic images if not available
try:
    import torch
    import torchvision
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Will create synthetic test images instead.")

# Create sample_images directory if it doesn't exist
os.makedirs('sample_images', exist_ok=True)

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def create_sample_images():
    """Create sample Fashion-MNIST test images"""
    
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping Fashion-MNIST dataset creation.")
        return False
    
    # Load Fashion-MNIST test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create sample images for each class
    samples_per_class = 3
    created_samples = {class_name: 0 for class_name in CLASS_NAMES}
    
    for i, (image, label) in enumerate(test_dataset):
        if i >= 1000:  # Limit search to first 1000 samples
            break
            
        class_name = CLASS_NAMES[label]
        
        # Skip if we already have enough samples for this class
        if created_samples[class_name] >= samples_per_class:
            continue
            
        # Convert tensor to PIL Image
        image_np = image.squeeze().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode='L')
        
        # Save image
        filename = f"{class_name.replace('/', '_')}_{created_samples[class_name] + 1}.png"
        filepath = os.path.join('sample_images', filename)
        image_pil.save(filepath)
        
        created_samples[class_name] += 1
        print(f"Created: {filename}")
        
        # Check if we have enough samples for all classes
        if all(count >= samples_per_class for count in created_samples.values()):
            break
    
    print(f"\nSample images created successfully!")
    print(f"Total images: {sum(created_samples.values())}")
    print(f"Images per class: {created_samples}")

def create_synthetic_test_images():
    """Create simple synthetic test images for basic testing"""
    
    # Create simple test patterns
    test_patterns = [
        ("test_pattern_1.png", create_checkerboard_pattern()),
        ("test_pattern_2.png", create_circle_pattern()),
        ("test_pattern_3.png", create_noise_pattern()),
        ("test_gradient.png", create_gradient_pattern()),
        ("test_stripes.png", create_stripes_pattern())
    ]
    
    for filename, pattern in test_patterns:
        filepath = os.path.join('sample_images', filename)
        Image.fromarray(pattern, mode='L').save(filepath)
        print(f"Created synthetic test image: {filename}")

def create_checkerboard_pattern():
    """Create a checkerboard pattern"""
    pattern = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            if (i // 4 + j // 4) % 2 == 0:
                pattern[i, j] = 255
    return pattern

def create_circle_pattern():
    """Create a circle pattern"""
    pattern = np.zeros((28, 28), dtype=np.uint8)
    center = 14
    radius = 8
    for i in range(28):
        for j in range(28):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                pattern[i, j] = 255
    return pattern

def create_noise_pattern():
    """Create a random noise pattern"""
    return np.random.randint(0, 256, (28, 28), dtype=np.uint8)

def create_gradient_pattern():
    """Create a gradient pattern"""
    pattern = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        pattern[i, :] = int(255 * i / 27)
    return pattern

def create_stripes_pattern():
    """Create a striped pattern"""
    pattern = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        if i % 4 < 2:
            pattern[i, :] = 255
    return pattern

if __name__ == "__main__":
    print("Creating sample images for Fashion-MNIST classifier...")
    
    # Always create synthetic test images first (these don't require PyTorch)
    print("Creating synthetic test images...")
    create_synthetic_test_images()
    
    # Try to create real Fashion-MNIST samples if PyTorch is available
    if PYTORCH_AVAILABLE:
        try:
            print("Creating Fashion-MNIST samples...")
            create_sample_images()
        except Exception as e:
            print(f"Could not create Fashion-MNIST samples: {e}")
    else:
        print("Skipping Fashion-MNIST samples (PyTorch not available)")
    
    print("\nSample image creation completed!")
    print("Images saved in 'sample_images/' directory")
    print("You can use these images to test the API endpoints.") 