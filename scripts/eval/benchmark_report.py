#!/usr/bin/env python3
"""
PPE Detection Benchmark Report Generator
Fetches training results, configs, and args from all YOLO runs and displays them in CLI.
No external dependencies - uses only Python stdlib.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Any


class SimpleYAMLParser:
    """Simple YAML parser for our use case"""
    @staticmethod
    def load(filepath: Path) -> Dict:
        result = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        # Parse value
                        if value.lower() == 'true':
                            result[key] = True
                        elif value.lower() == 'false':
                            result[key] = False
                        elif value == 'null':
                            result[key] = None
                        else:
                            try:
                                result[key] = float(value)
                                if result[key].is_integer():
                                    result[key] = int(result[key])
                            except:
                                result[key] = value
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        return result


class Table:
    """Simple ASCII table formatter"""
    @staticmethod
    def format(headers: List[str], rows: List[List[str]], title: str = "") -> str:
        """Format data as ASCII table"""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Build table
        lines = []

        # Title
        if title:
            total_width = sum(col_widths) + (len(headers) * 3) + 1
            lines.append("=" * total_width)
            lines.append(title.center(total_width))
            lines.append("=" * total_width)

        # Header separator
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines.append(sep)

        # Headers
        header_line = "|"
        for i, h in enumerate(headers):
            header_line += f" {h.ljust(col_widths[i])} |"
        lines.append(header_line)
        lines.append(sep)

        # Rows
        for row in rows:
            row_line = "|"
            for i, cell in enumerate(row):
                row_line += f" {str(cell).ljust(col_widths[i])} |"
            lines.append(row_line)

        lines.append(sep)
        return "\n".join(lines)


class BenchmarkReport:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs" / "detect"
        self.config_dir = self.project_root / "config"

    def get_model_name_from_args(self, args_path: Path) -> str:
        """Extract model name from args.yaml"""
        try:
            args = SimpleYAMLParser.load(args_path)
            return args.get('model', 'unknown').replace('.pt', '')
        except:
            return 'unknown'

    def get_best_metrics(self, results_csv: Path) -> Dict[str, float]:
        """Get metrics from the last epoch (best) in results.csv"""
        try:
            rows = []
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                last_row = rows[-1]
                return {
                    'epoch': int(float(last_row['epoch'])),
                    'precision': float(last_row['metrics/precision(B)']),
                    'recall': float(last_row['metrics/recall(B)']),
                    'mAP50': float(last_row['metrics/mAP50(B)']),
                    'mAP50-95': float(last_row['metrics/mAP50-95(B)']),
                }
        except Exception as e:
            pass
        return {}

    def get_model_size(self, weights_dir: Path) -> float:
        """Get best.pt file size in MB"""
        try:
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                return round(best_pt.stat().st_size / (1024 * 1024), 1)
        except:
            pass
        return 0.0

    def get_training_config(self, args_path: Path) -> Dict[str, Any]:
        """Extract training configuration from args.yaml"""
        try:
            args = SimpleYAMLParser.load(args_path)
            return {
                'epochs': args.get('epochs', 'N/A'),
                'batch': args.get('batch', 'N/A'),
                'imgsz': args.get('imgsz', 'N/A'),
                'optimizer': args.get('optimizer', 'N/A'),
                'lr0': args.get('lr0', 'N/A'),
                'patience': args.get('patience', 'N/A'),
                'device': args.get('device', 'N/A'),
            }
        except:
            return {}

    def get_dataset_info(self) -> Dict[str, int]:
        """Get dataset statistics"""
        dataset_path = self.project_root / "data" / "PPR_EXT.v2-raw.yolov11"
        stats = {}

        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split / "images"
            if split_path.exists():
                count = len(list(split_path.glob('*.*')))
                stats[split] = count

        return stats

    def get_class_info(self) -> Dict[str, Any]:
        """Get class information from data.yaml"""
        data_yaml = self.project_root / "data" / "PPR_EXT.v2-raw.yolov11" / "data.yaml"
        try:
            args = SimpleYAMLParser.load(data_yaml)
            # Extract names manually since it's a list
            classes = []
            with open(data_yaml, 'r') as f:
                for line in f:
                    if line.strip().startswith('names:'):
                        # Parse the list: ['class1', 'class2', ...]
                        names_str = line.split('names:', 1)[1].strip()
                        # Remove brackets and split by comma
                        names_str = names_str.strip('[]')
                        classes = [c.strip().strip("'\"") for c in names_str.split(',')]
                        break

            return {
                'num_classes': args.get('nc', 'N/A'),
                'classes': classes,
            }
        except:
            return {}

    def scan_runs(self) -> List[Dict[str, Any]]:
        """Scan runs directory and collect all training results"""
        results = []

        if not self.runs_dir.exists():
            print(f"Error: Runs directory not found at {self.runs_dir}")
            return results

        # Find all ppe_v2-* directories
        for run_dir in sorted(self.runs_dir.glob('ppe_v2-*')):
            if not run_dir.is_dir():
                continue

            args_path = run_dir / "args.yaml"
            results_csv = run_dir / "results.csv"
            weights_dir = run_dir / "weights"

            if not args_path.exists():
                continue

            model_name = self.get_model_name_from_args(args_path)
            metrics = self.get_best_metrics(results_csv) if results_csv.exists() else {}
            model_size = self.get_model_size(weights_dir)
            config = self.get_training_config(args_path)

            results.append({
                'run_dir': run_dir.name,
                'model': model_name,
                'path': str(run_dir),
                'metrics': metrics,
                'size_mb': model_size,
                'config': config,
            })

        return results

    def print_benchmark_table(self, runs: List[Dict[str, Any]]) -> None:
        """Print benchmark results as a table"""
        table_data = []
        for run in runs:
            metrics = run['metrics']
            table_data.append([
                run['run_dir'],
                run['model'],
                f"{metrics.get('epoch', 'N/A')}",
                f"{metrics.get('mAP50', 0):.3f}",
                f"{metrics.get('mAP50-95', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                f"{run['size_mb']} MB",
            ])

        headers = ["Run Folder", "Model", "Epochs", "mAP50", "mAP50-95", "Precision", "Recall", "Size"]
        print("\n" + Table.format(headers, table_data, "BENCHMARK RESULTS - ALL MODELS"))

    def print_training_config(self, runs: List[Dict[str, Any]]) -> None:
        """Print training configuration"""
        if runs:
            config = runs[0]['config']
            table_data = [
                ["Epochs", str(config.get('epochs', 'N/A'))],
                ["Batch Size", str(config.get('batch', 'N/A'))],
                ["Image Size", str(config.get('imgsz', 'N/A'))],
                ["Optimizer", str(config.get('optimizer', 'N/A'))],
                ["Initial LR", str(config.get('lr0', 'N/A'))],
                ["Patience (Early Stop)", str(config.get('patience', 'N/A'))],
                ["Device", str(config.get('device', 'N/A'))],
            ]
            headers = ["Parameter", "Value"]
            print("\n" + Table.format(headers, table_data, "TRAINING CONFIGURATION"))

    def print_dataset_info(self) -> None:
        """Print dataset statistics"""
        dataset = self.get_dataset_info()
        class_info = self.get_class_info()

        # Dataset split
        total = sum(dataset.values())
        dataset_table = [
            ["Train", str(dataset.get('train', 0)), f"{dataset.get('train', 0)/total*100:.1f}%"],
            ["Val", str(dataset.get('valid', 0)), f"{dataset.get('valid', 0)/total*100:.1f}%"],
            ["Test", str(dataset.get('test', 0)), f"{dataset.get('test', 0)/total*100:.1f}%"],
            ["TOTAL", str(total), "100.0%"],
        ]
        headers = ["Split", "Count", "Percentage"]
        print("\n" + Table.format(headers, dataset_table, "DATASET INFORMATION"))

        # Classes
        print("\nClasses:")
        if class_info.get('classes'):
            for idx, class_name in enumerate(class_info['classes']):
                print(f"  {idx}: {class_name}")

    def print_file_references(self, runs: List[Dict[str, Any]]) -> None:
        """Print file references for cross-checking"""
        ref_data = []
        for run in runs:
            run_path = Path(run['path'])
            ref_data.append([
                run['run_dir'],
                run['model'],
                str(run_path / "results.csv"),
            ])

        headers = ["Run", "Model", "Results CSV Path"]
        print("\n" + Table.format(headers, ref_data, "FILE REFERENCES FOR CROSS-CHECKING"))

    def print_summary(self, runs: List[Dict[str, Any]]) -> None:
        """Print summary and recommendations"""
        print("\n" + "="*100)
        print("SUMMARY & RECOMMENDATIONS".center(100))
        print("="*100 + "\n")

        # Find best model
        best_run = max(runs, key=lambda x: x['metrics'].get('mAP50', 0))

        print(f"✓ Models Trained: {len(runs)}")
        print(f"✓ Best Model: {best_run['model']} ({best_run['run_dir']})")
        print(f"  - mAP50: {best_run['metrics'].get('mAP50', 0):.3f}")
        print(f"  - mAP50-95: {best_run['metrics'].get('mAP50-95', 0):.3f}")
        print(f"  - Model Size: {best_run['size_mb']} MB")
        print(f"\n✓ Jetson Target: Orin Nano (standard)")
        print(f"✓ Recommended for deployment: {best_run['model']} (balance of accuracy & size)")

        # Find smallest model
        smallest = min(runs, key=lambda x: x['size_mb'])
        print(f"\n✓ Smallest Model: {smallest['model']} ({smallest['size_mb']} MB)")
        print(f"  - mAP50: {smallest['metrics'].get('mAP50', 0):.3f}")

    def generate_report(self) -> None:
        """Generate and print full report"""
        print("\n")
        print("█" * 100)
        print("PPE DETECTION - BENCHMARK REPORT".center(100, " "))
        print("█" * 100)

        runs = self.scan_runs()

        if not runs:
            print("No training runs found!")
            return

        self.print_benchmark_table(runs)
        self.print_training_config(runs)
        self.print_dataset_info()
        self.print_file_references(runs)
        self.print_summary(runs)

        print("\n" + "="*100 + "\n")


def main():
    # Get project root (script is in scripts/ folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    reporter = BenchmarkReport(str(project_root))
    reporter.generate_report()


if __name__ == "__main__":
    main()
