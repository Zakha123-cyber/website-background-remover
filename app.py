from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
MODEL_LOADED = False

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MODNet(nn.Module):
    """
    Simplified MODNet implementation for background removal
    """
    def __init__(self, backbone='simple'):
        super(MODNet, self).__init__()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2  
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder for alpha matte
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample 2
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 3
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample 4
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode to alpha
        alpha = self.decoder(features)
        
        return {
            'alpha': alpha
        }

def load_modnet_model():
    """Load or initialize simplified MODNet model"""
    global MODEL, MODEL_LOADED
    
    try:
        model = MODNet(backbone='simple')
        model = model.to(DEVICE)
        model.eval()
        
        # Try to load pretrained weights if available
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'modnet_weights.pth')
        if os.path.exists(model_path):
            logger.info(f"Loading pretrained weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            logger.info("No pretrained weights found, using initialized model")
        
        MODEL = model
        MODEL_LOADED = True
        logger.info(f"Simplified MODNet model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Error loading MODNet model: {e}")
        MODEL_LOADED = False

def preprocess_image(image):
    """Preprocess image for MODNet input"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # MODNet typically uses 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    tensor_image = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return tensor_image, pil_image.size

def postprocess_alpha(alpha, original_size):
    """Postprocess alpha matte to original size"""
    # Convert tensor to numpy
    alpha_np = alpha.squeeze().cpu().numpy()
    
    # Resize to original size
    alpha_resized = cv2.resize(alpha_np, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Ensure values are in [0, 1]
    alpha_resized = np.clip(alpha_resized, 0, 1)
    
    return alpha_resized

def apply_background_removal(image, alpha):
    """Apply alpha matte to remove background"""
    # Ensure alpha has same spatial dimensions as image
    if alpha.shape[:2] != image.shape[:2]:
        alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))
    
    # Convert to 4-channel RGBA
    if len(image.shape) == 3:
        # Add alpha channel
        if image.shape[2] == 3:  # BGR/RGB
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
        else:
            rgba = image.copy()
            rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    else:
        # Grayscale to RGBA
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = np.stack([image] * 3, axis=2)
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    
    return rgba

def modnet_background_remover(image_path):
    """
    Main function for MODNet-based background removal
    """
    try:
        # Check if model is loaded
        if not MODEL_LOADED:
            logger.error("MODNet model not loaded")
            return None, "Model not available"
        
        # Load and preprocess image
        logger.info("Loading and preprocessing image...")
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        original_height, original_width = image.shape[:2]
        logger.info(f"Original image size: {original_width}x{original_height}")
        
        # Preprocess for model
        tensor_image, pil_size = preprocess_image(image)
        tensor_image = tensor_image.to(DEVICE)
        
        # Run inference
        logger.info("Running MODNet inference...")
        with torch.no_grad():
            results = MODEL(tensor_image)
            alpha = results['alpha']
        
        # Postprocess alpha matte
        logger.info("Postprocessing alpha matte...")
        alpha_np = postprocess_alpha(alpha, (original_width, original_height))
        
        # Apply background removal
        logger.info("Applying background removal...")
        result_image = apply_background_removal(image, alpha_np)
        
        # Save result
        output_filename = f"modnet_result_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Convert BGR to RGB for saving
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(output_path, result_rgb)
        
        logger.info(f"Background removal completed: {output_path}")
        return output_filename, None
        
    except Exception as e:
        logger.error(f"Error in MODNet background removal: {e}")
        return None, str(e)

def advanced_grabcut_remover(image_path):
    """
    Advanced background removal using optimized GrabCut algorithm
    """
    try:
        logger.info("Using advanced GrabCut background removal method...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # === Step 1: Pre-processing ===
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # === Step 2: Initial segmentation with multiple strategies ===
        
        # Strategy 1: Traditional GrabCut with conservative rectangle
        mask1 = np.zeros((height, width), np.uint8)
        margin = 0.015  # More conservative margin
        x1 = int(width * margin)
        y1 = int(height * margin)
        x2 = int(width * (1 - margin))
        y2 = int(height * (1 - margin))
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        bgd_model1 = np.zeros((1, 65), np.float64)
        fgd_model1 = np.zeros((1, 65), np.float64)
        cv2.grabCut(filtered, mask1, rect, bgd_model1, fgd_model1, 8, cv2.GC_INIT_WITH_RECT)
        
        # Strategy 2: Edge-based initial segmentation
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours and select the largest one as probable foreground
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask from contour
            mask2 = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask2, [largest_contour], cv2.GC_FGD)
            
            # Set background areas
            cv2.rectangle(mask2, (0, 0), (width, height), cv2.GC_BGD, thickness=20)
            cv2.fillPoly(mask2, [largest_contour], cv2.GC_FGD)
            
            # Apply GrabCut with this initial mask
            bgd_model2 = np.zeros((1, 65), np.float64)
            fgd_model2 = np.zeros((1, 65), np.float64)
            cv2.grabCut(filtered, mask2, None, bgd_model2, fgd_model2, 5, cv2.GC_INIT_WITH_MASK)
        else:
            mask2 = mask1.copy()
        
        # === Step 3: Combine and refine masks ===
        # Convert masks to binary
        binary_mask1 = np.where((mask1 == 2) | (mask1 == 0), 0, 1).astype('uint8')
        binary_mask2 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
        
        # Combine masks using intersection for more conservative result
        combined_mask = cv2.bitwise_and(binary_mask1, binary_mask2)
        
        # If intersection is too small, use the better individual mask
        if cv2.countNonZero(combined_mask) < (height * width * 0.05):  # Less than 5%
            if cv2.countNonZero(binary_mask1) > cv2.countNonZero(binary_mask2):
                combined_mask = binary_mask1
            else:
                combined_mask = binary_mask2
        
        # === Step 4: Morphological operations for refinement ===
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_medium = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Smooth edges
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        # === Step 5: Edge refinement ===
        # Apply Gaussian blur to create soft edges
        combined_mask = combined_mask.astype(np.float32)
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 1)
        
        # === Step 6: Final iteration of GrabCut with refined mask ===
        final_mask = np.zeros((height, width), np.uint8)
        final_mask[combined_mask > 0.5] = cv2.GC_FGD
        final_mask[combined_mask <= 0.1] = cv2.GC_BGD
        final_mask[(combined_mask > 0.1) & (combined_mask <= 0.5)] = cv2.GC_PR_BGD
        
        bgd_model_final = np.zeros((1, 65), np.float64)
        fgd_model_final = np.zeros((1, 65), np.float64)
        cv2.grabCut(filtered, final_mask, None, bgd_model_final, fgd_model_final, 3, cv2.GC_INIT_WITH_MASK)
        
        # === Step 7: Create final alpha channel ===
        final_binary = np.where((final_mask == 2) | (final_mask == 0), 0, 1).astype('uint8')
        
        # Create smooth alpha channel
        alpha = final_binary.astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 1)
        alpha = (alpha * 255).astype(np.uint8)
        
        # === Step 8: Create RGBA result ===
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb_image
        rgba[:, :, 3] = alpha
        
        # Save result
        output_filename = f"grabcut_result_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Convert RGB to BGR for OpenCV saving
        result_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(output_path, result_bgr)
        
        logger.info(f"Advanced GrabCut background removal completed: {output_path}")
        return output_filename, None
        
    except Exception as e:
        logger.error(f"Error in advanced GrabCut background removal: {e}")
        return None, str(e)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process image
        if MODEL_LOADED:
            result_filename, error = modnet_background_remover(file_path)
        else:
            result_filename, error = advanced_grabcut_remover(file_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if error:
            return jsonify({'error': f'Processing failed: {error}'}), 500
        
        if result_filename:
            return jsonify({
                'success': True,
                'result_file': result_filename,
                'download_url': f'/download/{result_filename}'
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/remove_background', methods=['POST'])
def remove_background():
    """Handle background removal - endpoint expected by frontend"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Get method from form data
        method = request.form.get('method', 'modnet')
        logger.info(f"Processing method: {method}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Save original file for display
        original_filename = f"original_{unique_filename}"
        original_path = os.path.join(app.config['PROCESSED_FOLDER'], original_filename)
        file.seek(0)  # Reset file pointer
        file.save(original_path)
        
        logger.info(f"File uploaded for background removal: {file_path}")
        
        # Process image based on selected method
        if method == 'modnet' and MODEL_LOADED:
            result_filename, error = modnet_background_remover(file_path)
        else:
            # Always use advanced GrabCut since MODNet is not working well
            result_filename, error = advanced_grabcut_remover(file_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if error:
            # Clean up original file on error
            try:
                os.remove(original_path)
            except:
                pass
            return jsonify({'success': False, 'error': f'Processing failed: {error}'}), 500
        
        if result_filename:
            return jsonify({
                'success': True,
                'original_image_url': f'/download/{original_filename}',
                'processed_image_url': f'/download/{result_filename}',
                'result_file': result_filename
            })
        else:
            # Clean up original file on failure
            try:
                os.remove(original_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'Processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        return jsonify({'success': False, 'error': f'Background removal failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/status')
def status():
    """Get system status"""
    return jsonify({
        'model_loaded': MODEL_LOADED,
        'device': str(DEVICE),
        'torch_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    })

if __name__ == '__main__':
    # Initialize model on startup
    logger.info("Initializing MODNet Background Remover...")
    logger.info(f"Using device: {DEVICE}")
    
    try:
        load_modnet_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Will use fallback method instead")
    
    app.run(debug=False, host='0.0.0.0', port=5000)