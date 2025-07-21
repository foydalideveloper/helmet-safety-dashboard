# app.py - Enterprise-Grade Safety Monitoring System
import gradio as gr
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import hashlib
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time

# --- 1. CONFIGURATION & CORE CLASSES ---

@dataclass
class AppConfig:
    MODEL_PATH: str = 'best.pt'
    SNAPSHOT_DIR: str = "violation_snapshots"
    DATABASE_PATH: str = "safety_monitoring.db"
    MAX_SNAPSHOTS: int = 50
    ALERT_THRESHOLD: int = 5  # violations per minute

@dataclass
class Violation:
    timestamp: datetime
    frame_number: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_name: str
    zone_id: Optional[int] = None
    severity: str = "medium"

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, video_hash TEXT, timestamp TEXT, 
                    frame_number INTEGER, confidence REAL, bbox TEXT, class_name TEXT, 
                    zone_id INTEGER, severity TEXT
                )
            ''')
            conn.commit()

    def save_violations(self, violations: List[Violation], video_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for v in violations:
                cursor.execute(
                    "INSERT INTO violations (video_hash, timestamp, frame_number, confidence, bbox, class_name, zone_id, severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (video_hash, v.timestamp.isoformat(), v.frame_number, v.confidence, json.dumps(v.bbox), v.class_name, v.zone_id, v.severity)
                )
            conn.commit()

    def get_violation_history(self, days: int = 30) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            query = "SELECT * FROM violations WHERE timestamp >= ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        return df

class SafetyMonitor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = YOLO(config.MODEL_PATH)
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        Path(config.SNAPSHOT_DIR).mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Model loaded. Classes: {self.model.names}")

    def _calculate_video_hash(self, video_path: str) -> str:
        hasher = hashlib.md5()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""): hasher.update(chunk)
        return hasher.hexdigest()

    def _is_in_zone(self, bbox: Tuple[int, int, int, int], zones: List[Tuple[int, int, int, int]]) -> Optional[int]:
        if not zones: return None
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
            if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                return i + 1  # Return 1-based zone ID
        return None
    
    def _determine_severity(self, confidence: float) -> str:
        if confidence > 0.8: return "High"
        if confidence > 0.6: return "Medium"
        return "Low"

    def analyze_video(self, video_path: str, conf_threshold: float, detection_zones: List, progress=gr.Progress()):
        start_time = time.time()
        logging.info(f"Starting analysis for {video_path}")
        for f in Path(self.config.SNAPSHOT_DIR).glob("*.jpg"): f.unlink()

        video_hash = self._calculate_video_hash(video_path)
        cap = cv2.VideoCapture(video_path)
        
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = "detection_output.mp4"
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        all_violations = []
        snapshot_paths = []
        violation_timeline = defaultdict(int)

        for frame_num in progress.tqdm(range(total_frames), desc="Analyzing Video"):
            success, frame = cap.read()
            if not success: break

            results = self.model.track(frame, persist=True, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            for zone_idx, (x1, y1, x2, y2) in enumerate(detection_zones):
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Zone {zone_idx+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            frame_violations_count = 0
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names.get(class_id, "unknown")
                    
                    if class_name == 'head':
                        bbox = tuple(map(int, box.xyxy[0].tolist()))
                        violation = Violation(
                            timestamp=datetime.now(), frame_number=frame_num,
                            confidence=float(box.conf), bbox=bbox, class_name=class_name,
                            zone_id=self._is_in_zone(bbox, detection_zones),
                            severity=self._determine_severity(float(box.conf))
                        )
                        all_violations.append(violation)
                        frame_violations_count += 1
            
            if frame_violations_count > 0 and len(snapshot_paths) < self.config.MAX_SNAPSHOTS:
                snapshot_path = Path(self.config.SNAPSHOT_DIR) / f"violation_frame_{frame_num}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                snapshot_paths.append(str(snapshot_path))
            
            minute_mark = int((frame_num / fps) / 60)
            violation_timeline[minute_mark] += frame_violations_count
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        if all_violations:
            self.db_manager.save_violations(all_violations, video_hash)
            
        processing_time = time.time() - start_time
        video_duration = total_frames / fps
        stats = {
            "Analysis Summary": {"Processing Time (s)": f"{processing_time:.2f}", "Video Duration (s)": f"{video_duration:.2f}"},
            "Violation Stats": {"Total Violations": len(all_violations), "Avg Confidence": f"{np.mean([v.confidence for v in all_violations]):.2f}" if all_violations else "N/A"},
            "Severity Breakdown": dict(pd.Series([v.severity for v in all_violations]).value_counts()) if all_violations else {},
            "Zone Breakdown": dict(pd.Series([f'Zone {v.zone_id}' for v in all_violations if v.zone_id]).value_counts()) if any(v.zone_id for v in all_violations) else {}
        }
        alerts = [f"High violation rate in minute {m}: {c} violations" for m, c in violation_timeline.items() if c >= self.config.ALERT_THRESHOLD]
        
        logging.info(f"Analysis complete. Found {len(all_violations)} violations.")
        return output_video_path, stats, snapshot_paths, dict(violation_timeline), alerts

# --- 2. GRADIO UI FUNCTIONS ---

def create_chart(data: Dict, title: str, x_axis: str, y_axis: str):
    if not data: return go.Figure().add_annotation(text="No Data Available", showarrow=False)
    df = pd.DataFrame(list(data.items()), columns=[x_axis, y_axis])
    if df.empty: return go.Figure().add_annotation(text="No Data Available", showarrow=False)
    
    if y_axis == 'Severity':
        fig = px.pie(df, names=x_axis, values=y_axis, title=title)
    else:
        fig = px.bar(df, x=x_axis, y=y_axis, title=title)
    return fig

def main_interface_function(video_path, conf_threshold, zones_str, progress=gr.Progress()):
    if video_path is None: return None, None, None, None, None, "Please upload a video first.", "Status: Idle"
    
    try:
        zones = []
        if zones_str.strip():
            for line in zones_str.strip().split('\n'):
                coords = [int(c.strip()) for c in line.split(',')]
                if len(coords) == 4: zones.append(tuple(coords))
        
        monitor = SafetyMonitor(AppConfig())
        video_out, stats, snapshots, timeline, alerts = monitor.analyze_video(video_path, conf_threshold, zones, progress)
        
        timeline_chart = create_chart(timeline, "Violation Timeline", "Minute", "Violations")
        severity_chart = create_chart(stats.get("Severity Breakdown", {}), "Severity Distribution", "Severity", "Count")
        
        alerts_text = "\n".join(f"⚠️ {alert}" for alert in alerts) if alerts else "✅ No alerts detected."
        status_text = f"Processing complete. Found {stats['Violation Stats']['Total Violations']} violations."
        
        return video_out, stats, snapshots, timeline_chart, severity_chart, alerts_text, status_text
    except Exception as e:
        logging.error(f"UI function error: {e}", exc_info=True)
        error_msg = f"An error occurred: {e}"
        empty_fig = go.Figure()
        return None, {"Error": error_msg}, [], empty_fig, empty_fig, error_msg, "Status: Error"

def get_historical_data_interface():
    try:
        db = DatabaseManager(AppConfig().DATABASE_PATH)
        df = db.get_violation_history(30)
        if df.empty: return "No historical data in the last 30 days.", go.Figure()

        summary = f"**Total Violations (30d):** {len(df)}\n\n**Most Common Severity:** {df['severity'].mode()[0]}"
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        fig = px.line(daily_counts, x='date', y='count', title="Daily Violation Trend", markers=True)
        return summary, fig
    except Exception as e:
        logging.error(f"Historical data error: {e}", exc_info=True)
        return f"Error: {e}", go.Figure()

# --- 3. GRADIO APPLICATION LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="Enterprise Safety Monitoring") as demo:
    gr.HTML("<h1>🛡️ Enterprise Safety Monitoring System</h1><p>Advanced AI-Powered Workplace Safety Analysis</p>")
    
    with gr.Tabs():
        with gr.Tab("🔍 Live Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video for Analysis")
                    with gr.Accordion("⚙️ Analysis Settings", open=True):
                        conf_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence Threshold")
                        detection_zones_input = gr.Textbox(label="Detection Zones (x1,y1,x2,y2 per line)", lines=3, placeholder="100,100,500,400\n600,200,900,500")
                    run_btn = gr.Button("🚀 Start Full Analysis", variant="primary")
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Processed Video Output")
                    status_output = gr.Textbox(label="Processing Status", interactive=False)
        
        with gr.Tab("📊 Results & Evidence"):
            with gr.Row():
                stats_output = gr.JSON(label="Detailed Statistics")
                alerts_output = gr.Textbox(label="🚨 Safety Alerts", interactive=False)
            with gr.Row():
                timeline_chart = gr.Plot(label="Violation Timeline")
                severity_chart = gr.Plot(label="Severity Distribution")
            snapshot_gallery = gr.Gallery(label="📸 Violation Snapshots", columns=6, height="auto")

        with gr.Tab("📈 Historical Data"):
            refresh_history_btn = gr.Button("🔄 Refresh Historical Data", variant="secondary")
            history_summary = gr.Markdown()
            history_chart = gr.Plot()

    # Define component interactions
    run_btn.click(
        fn=main_interface_function,
        inputs=[video_input, conf_slider, detection_zones_input],
        outputs=[video_output, stats_output, snapshot_gallery, timeline_chart, severity_chart, alerts_output, status_output]
    )
    refresh_history_btn.click(fn=get_historical_data_interface, outputs=[history_summary, history_chart])

    gr.Examples(
        examples=[["30sec.mp4", 0.4, "0,0,320,360\n320,0,640,360"]],
        inputs=[video_input, conf_slider, detection_zones_input],
    )

if __name__ == "__main__":
    demo.launch(debug=True)