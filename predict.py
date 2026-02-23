#!/usr/bin/env python3
"""
Improved Multi-Label Prediction with Per-Class Thresholds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np

from src.models import create_improved_model

DISEASE_INFO = {
    'Disease_Risk': ('Disease Risk', 'General indicator of eye disease presence'),
    'DR': ('Diabetic Retinopathy', 'Diabetes-related retinal blood vessel damage'),
    'ARMD': ('Age-related Macular Degeneration', 'Central vision loss in elderly'),
    'MH': ('Macular Hole', 'Small break in the macula'),
    'DN': ('Diabetic Neuropathy', 'Diabetes-related nerve damage'),
    'MYA': ('Myopia', 'Nearsightedness with retinal changes'),
    'BRVO': ('Branch Retinal Vein Occlusion', 'Blocked branch retinal vein'),
    'TSLN': ('Tessellation', 'Visible choroidal vessels'),
    'ERM': ('Epiretinal Membrane', 'Scar tissue on retina'),
    'LS': ('Laser Scars', 'Previous laser treatment marks'),
    'MS': ('Myelinated Nerve Fibers', 'White patches near optic disc'),
    'CSR': ('Central Serous Retinopathy', 'Fluid under central retina'),
    'ODC': ('Optic Disc Cupping', 'Glaucoma indicator'),
    'CRVO': ('Central Retinal Vein Occlusion', 'Blocked central retinal vein'),
    'AH': ('Asteroid Hyalosis', 'Calcium deposits in vitreous'),
    'ODP': ('Optic Disc Pallor', 'Pale optic disc'),
    'ODE': ('Optic Disc Edema', 'Swollen optic disc'),
    'AION': ('Anterior Ischemic Optic Neuropathy', 'Sudden optic nerve blood supply loss'),
    'PT': ('Phthisis', 'Shrunken non-functional eye'),
    'RT': ('Retinitis', 'Retinal inflammation'),
    'RS': ('Retinal Scars', 'Scarred retinal tissue'),
    'CRS': ('Chorioretinal Scars', 'Deep scars in retina'),
    'EDN': ('Exudative Diabetic Neuropathy', 'Fluid leakage from diabetic nerve damage'),
    'RPEC': ('RPE Changes', 'Retinal pigment epithelium changes'),
    'MHL': ('Macular Hole Lamellar', 'Partial-thickness macular defect'),
    'CATARACT': ('Cataract', 'Clouding of the eye lens'),
    'GLAUCOMA': ('Glaucoma', 'Optic nerve damage'),
    'NORMAL': ('Normal', 'No significant pathology detected'),
    'RD': ('Retinal Detachment', 'Retina pulls away from supportive tissue'),
    'RP': ('Retinitis Pigmentosa', 'Inherited retinal disorder causing vision loss'),
}



class ImprovedMultiLabelClassifier:
    """
    Multi-label classifier with per-class optimized thresholds,
    temperature scaling, and top-K prediction constraints.
    """

    def __init__(
            self,
            checkpoint_path: str = "models/ocunetv4.pth",
            config_path: str = "config/config.yaml",
            thresholds_path: str = "evaluation_results/optimal_thresholds.yaml",
            calibration_path: str = "evaluation_results/calibration.yaml",
            device: Optional[str] = None,
            max_predictions: int = 5,
            min_confidence: float = 0.3
    ):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prediction constraints
        self.max_predictions = max_predictions
        self.min_confidence = min_confidence

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint.get('class_names', [])
        self.num_classes = len(self.class_names)
        self.default_threshold = checkpoint.get('threshold', 0.5)

        # Load calibration (temperature scaling) if available
        self.temperature = 1.0
        cal_file = Path(calibration_path)
        if cal_file.exists():
            with open(cal_file, 'r') as f:
                cal_data = yaml.safe_load(f)
            self.temperature = cal_data.get('temperature', 1.0)
            # Use calibrated thresholds from calibration file
            self.thresholds = {}
            cal_thresholds = cal_data.get('thresholds', {})
            for name in self.class_names:
                if name in cal_thresholds:
                    self.thresholds[name] = cal_thresholds[name]['threshold']
                else:
                    self.thresholds[name] = self.default_threshold
            print(f"Loaded calibration: T={self.temperature:.4f}")
        else:
            # Fall back to optimal_thresholds.yaml
            self.thresholds = {}
            thresholds_file = Path(thresholds_path)
            if thresholds_file.exists():
                with open(thresholds_file, 'r') as f:
                    thresholds_data = yaml.safe_load(f)
                    for name, data in thresholds_data.items():
                        self.thresholds[name] = data['threshold']
                print(f"Loaded optimized thresholds from {thresholds_path}")
            else:
                for name in self.class_names:
                    self.thresholds[name] = self.default_threshold
                print(f"Using default threshold: {self.default_threshold}")

        # Create model
        self.model = create_improved_model(self.config, self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Transform (respects config image_size)
        img_size = self.config.get('dataset', {}).get('image_size', 224)

        # Include preprocessing if available
        preprocess_list = []
        try:
            from src.preprocessing import FundusROICrop, CLAHETransform
            preprocess_config = self.config.get('preprocessing', {})
            if preprocess_config.get('fundus_roi_crop', False):
                preprocess_list.append(FundusROICrop())
            if preprocess_config.get('clahe', False):
                preprocess_list.append(CLAHETransform(
                    clip_limit=preprocess_config.get('clahe_clip_limit', 2.0)
                ))
        except ImportError:
            pass

        self.transform = transforms.Compose(preprocess_list + [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded on {self.device}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Temperature: {self.temperature:.4f}")
        print(f"  Max predictions: {self.max_predictions}")
        print(f"  Min confidence: {self.min_confidence}")

    def predict(
            self,
            image_path: str,
            use_optimized_thresholds: bool = True
    ) -> Dict:
        """
        Predict diseases from retinal image.

        Args:
            image_path: Path to image
            use_optimized_thresholds: Use per-class optimized thresholds

        Returns:
            Dict with predictions
        """
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.sigmoid(scaled_logits).squeeze().cpu().numpy()

        # Apply thresholds
        detected = []
        probabilities = {}
        all_predictions = {}

        for i, name in enumerate(self.class_names):
            prob = float(probs[i])

            if use_optimized_thresholds:
                threshold = self.thresholds.get(name, self.default_threshold)
            else:
                threshold = self.default_threshold

            is_detected = prob >= threshold and prob >= self.min_confidence

            probabilities[name] = prob

            info = DISEASE_INFO.get(name, (name, 'No description'))

            all_predictions[name] = {
                'detected': bool(is_detected),
                'probability': prob,
                'threshold': threshold,
                'full_name': info[0],
                'description': info[1],
                'confidence': 'HIGH' if prob >= 0.8 else 'MEDIUM' if prob >= 0.5 else 'LOW'
            }

            if is_detected:
                detected.append(name)

        # Sort by probability
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
        detected = sorted(detected, key=lambda x: probabilities[x], reverse=True)

        # Top-K constraint: keep only top max_predictions
        if len(detected) > self.max_predictions:
            detected = detected[:self.max_predictions]
            # Update all_predictions to reflect the constraint
            for name in self.class_names:
                if name not in detected:
                    all_predictions[name]['detected'] = False

        return {
            'detected_diseases': detected,
            'num_diseases': len(detected),
            'probabilities': probabilities,
            'all_predictions': all_predictions,
            'image_path': image_path,
            'temperature': self.temperature
        }

    def predict_batch(
            self,
            image_paths: List[str],
            use_optimized_thresholds: bool = True
    ) -> List[Dict]:
        """Batch prediction."""
        return [self.predict(p, use_optimized_thresholds) for p in image_paths]

    def format_result(self, result: Dict, verbose: bool = True) -> str:
        """Format result as string."""
        lines = [
            "=" * 70,
            "OCUNET MULTI-LABEL DIAGNOSIS",
            "=" * 70,
            f"Image: {Path(result['image_path']).name}",
            "",
            f"DETECTED CONDITIONS: {result['num_diseases']}",
            "-" * 50,
        ]

        if result['detected_diseases']:
            for disease in result['detected_diseases']:
                pred = result['all_predictions'][disease]
                conf_emoji = "🔴" if pred['confidence'] == 'HIGH' else "🟡" if pred['confidence'] == 'MEDIUM' else "🟢"

                lines.append(f"{conf_emoji} {pred['full_name']} ({disease})")
                lines.append(f"   Probability: {pred['probability']:.1%} (threshold: {pred['threshold']:.0%})")

                if verbose:
                    lines.append(f"   {pred['description']}")
                lines.append("")
        else:
            lines.append("  ✓ No significant pathology detected")
            lines.append("")

        # Top probabilities
        lines.extend([
            "-" * 50,
            "TOP 5 PROBABILITIES:",
            "-" * 50,
        ])

        for i, (disease, prob) in enumerate(list(result['probabilities'].items())[:5]):
            threshold = self.thresholds.get(disease, self.default_threshold)
            status = "✓" if prob >= threshold else " "
            info = DISEASE_INFO.get(disease, (disease, ''))[0]
            lines.append(f"  {status} {info}: {prob:.1%}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def generate_report(self, result: Dict, output_path: str = None) -> str:
        """Generate detailed medical report."""

        report_lines = [
            "=" * 70,
            "OCUNET - RETINAL DISEASE SCREENING REPORT",
            "=" * 70,
            "",
            "PATIENT INFORMATION",
            "-" * 50,
            f"Image: {Path(result['image_path']).name}",
            f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "FINDINGS",
            "-" * 50,
        ]

        if result['detected_diseases']:
            report_lines.append(f"Number of conditions detected: {result['num_diseases']}")
            report_lines.append("")

            for i, disease in enumerate(result['detected_diseases'], 1):
                pred = result['all_predictions'][disease]

                report_lines.extend([
                    f"{i}. {pred['full_name']} ({disease})",
                    f"   - Probability: {pred['probability']:.1%}",
                    f"   - Confidence: {pred['confidence']}",
                    f"   - Description: {pred['description']}",
                    ""
                ])
        else:
            report_lines.extend([
                "No significant pathology detected.",
                "The retinal image appears normal.",
                ""
            ])

        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 50,
        ])

        # Generate recommendations based on findings
        if 'DR' in result['detected_diseases'] or 'DN' in result['detected_diseases']:
            report_lines.append("• Urgent: Diabetic eye disease detected. Recommend immediate ophthalmology referral.")
            report_lines.append("• Monitor blood glucose levels closely.")

        if 'GLAUCOMA' in result['detected_diseases'] or 'ODC' in result['detected_diseases']:
            report_lines.append("• Glaucoma indicators detected. Recommend IOP measurement and visual field testing.")

        if 'CATARACT' in result['detected_diseases']:
            report_lines.append("• Cataract detected. Consider surgical evaluation if vision is affected.")

        if 'ARMD' in result['detected_diseases']:
            report_lines.append(
                "• Age-related macular degeneration detected. Recommend AREDS supplements and regular monitoring.")

        if 'RD' in result['detected_diseases']:
            report_lines.append("• EMERGENCY: Retinal Detachment detected. Immediate ophthalmological intervention required.")

        if 'RP' in result['detected_diseases']:
            report_lines.append("• Retinitis Pigmentosa indicators detected. Recommend comprehensive retinal evaluation.")

        if not result['detected_diseases']:
            report_lines.append("• Continue routine screening schedule.")
            report_lines.append("• No immediate intervention required.")

        report_lines.extend([
            "",
            "DISCLAIMER",
            "-" * 50,
            "This report is generated by an AI system and is intended for",
            "clinical decision support only. It should not replace professional",
            "medical diagnosis. Please consult with a qualified ophthalmologist",
            "for definitive diagnosis and treatment recommendations.",
            "",
            "=" * 70,
        ])

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved: {output_path}")

        return report


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--report]")
        sys.exit(1)

    image_path = sys.argv[1]
    generate_report = '--report' in sys.argv

    # Initialize
    classifier = ImprovedMultiLabelClassifier()

    # Predict
    result = classifier.predict(image_path)

    # Output
    print(classifier.format_result(result))

    if generate_report:
        report_path = Path(image_path).stem + "_report.txt"
        classifier.generate_report(result, report_path)


if __name__ == "__main__":
    main()