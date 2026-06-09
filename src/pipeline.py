import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from mobilenetv3_custom import MobileNetV2MultiTask  

# Import model của bạn
# Ví dụ nếu file kiến trúc tên là model.py:
# from model import MobileNetV2MultiTask

# Nếu bạn đang để class MobileNetV2MultiTask trong cùng file này thì bỏ import trên.


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



DISEASE_LABELS = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
]

SEVERITY_LABELS = [
    "severity_0",
    "severity_1",
    "severity_2",
    "severity_3",
]



def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])


def load_image(image_path, img_size=224):
    pil_img = Image.open(image_path).convert("RGB")
    transform = build_transform(img_size)

    input_tensor = transform(pil_img).unsqueeze(0)

    # ảnh gốc resize để overlay Grad-CAM
    original_rgb = np.array(pil_img.resize((img_size, img_size)))
    return input_tensor, original_rgb




def clean_state_dict(state_dict):
    """
    Xử lý checkpoint nếu được train bằng DataParallel,
    key thường có prefix 'module.'.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(model, ckpt_path, device=DEVICE):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, task="disease", class_idx=None):
        """
        task:
            - "disease"
            - "severity"

        class_idx:
            - None: lấy class model dự đoán
            - int: ép Grad-CAM theo class cụ thể
        """
        self.model.zero_grad()

        disease_logits, severity_logits = self.model(input_tensor)

        if task == "disease":
            logits = disease_logits
        elif task == "severity":
            logits = severity_logits
        else:
            raise ValueError("task phải là 'disease' hoặc 'severity'.")

        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # gradients: [B, C, H, W]
        # activations: [B, C, H, W]
        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx, probs.detach().cpu().numpy()[0]



def overlay_cam_on_image(rgb_img, cam, alpha=0.45):
    """
    rgb_img: numpy RGB, shape [H, W, 3]
    cam: numpy, shape [h, w], value 0..1
    """
    h, w, _ = rgb_img.shape

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * alpha + rgb_img * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay



def predict_and_gradcam(
    image_path,
    checkpoint_path,
    output_dir="Grad-CAM",
    img_size=224,
    task_for_cam="disease"
):
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo model đúng với lúc train
    model = MobileNetV2MultiTask(
        num_disease_classes=5,
        num_severity_classes=4
    )

    model = load_checkpoint(model, checkpoint_path, DEVICE)

    input_tensor, original_rgb = load_image(image_path, img_size)
    input_tensor = input_tensor.to(DEVICE)

    # Layer nên dùng Grad-CAM:
    # Với MobileNetV2, thường lấy layer convolution cuối trong self.features.
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    # Forward để lấy prediction
    with torch.no_grad():
        disease_logits, severity_logits = model(input_tensor)

        disease_probs = F.softmax(disease_logits, dim=1)[0]
        severity_probs = F.softmax(severity_logits, dim=1)[0]

        disease_idx = torch.argmax(disease_probs).item()
        severity_idx = torch.argmax(severity_probs).item()

    disease_label = DISEASE_LABELS[disease_idx]
    severity_label = SEVERITY_LABELS[severity_idx]

    disease_conf = disease_probs[disease_idx].item()
    severity_conf = severity_probs[severity_idx].item()

    print("Disease prediction:")
    print(f"  class: {disease_idx} - {disease_label}")
    print(f"  confidence: {disease_conf:.4f}")

    print("Severity prediction:")
    print(f"  class: {severity_idx} - {severity_label}")
    print(f"  confidence: {severity_conf:.4f}")

    # Grad-CAM
    cam, cam_class_idx, cam_probs = gradcam(
        input_tensor,
        task=task_for_cam,
        class_idx=None
    )

    overlay = overlay_cam_on_image(original_rgb, cam)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cam_path = os.path.join(output_dir, f"{base_name}_gradcam_{task_for_cam}.jpg")

    Image.fromarray(overlay).save(cam_path)

    gradcam.remove_hooks()

    result = {
        "disease_idx": disease_idx,
        "disease_label": disease_label,
        "disease_confidence": disease_conf,
        "severity_idx": severity_idx,
        "severity_label": severity_label,
        "severity_confidence": severity_conf,
        "gradcam_task": task_for_cam,
        "gradcam_class_idx": cam_class_idx,
        "gradcam_path": cam_path,
    }

    return result



if __name__ == "__main__":
    image_path = r"data\Durian_Leaf_Diseases\test\algal\DLDD_TEST_000006.jpg"
    checkpoint_path = r"checkpoints\v2batch16_best\best_mobilenetv3_multitask.pth"

    result = predict_and_gradcam(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        output_dir="Grad-CAM",
        img_size=224,
        task_for_cam="severity",   # hoặc "severity"
    )

    print(result)

