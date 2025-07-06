#!/usr/bin/env python3
"""
USB Camera Video Capture with Terminal Control
Optimized version with raw terminal input
"""

import cv2
import time
import pathlib
import threading
import sys
import os
import tty
import termios
from datetime import datetime

# Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
VIDEO_DIR = "./videos"


class TerminalInput:
    """Handle raw terminal input in a separate thread"""
    
    def __init__(self):
        self.running = True
        self.thread = None
        self.current_key = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start input thread"""
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop input thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
            
    def _read_loop(self):
        """Read keyboard input without blocking"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            while self.running:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    with self.lock:
                        self.current_key = ch
                time.sleep(0.01)
        except Exception as e:
            print(f"\nInput error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
    def get_key(self):
        """Get and consume the current key press"""
        with self.lock:
            key = self.current_key
            self.current_key = None
            return key


class CameraCapture:
    """Simplified camera capture system"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None
        self.video_writer = None
        self.is_recording = False
        self.show_preview = False
        self.frame_count = 0
        self.recording_start = None
        self.input_handler = TerminalInput()
        self.last_status_time = 0
        self.last_status_text = ""
        
        # Create video directory
        pathlib.Path(VIDEO_DIR).mkdir(exist_ok=True, parents=True)
        
    def init_camera(self):
        """Initialize camera if not already initialized"""
        if self.camera is not None:
            return True
            
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            self.camera = None
            return False
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Test read
        ret, _ = self.camera.read()
        if not ret:
            self.camera.release()
            self.camera = None
            return False
            
        return True
        
    def release_camera(self):
        """Release camera if initialized"""
        if self.camera:
            self.camera.release()
            self.camera = None
            
    def toggle_recording(self, with_preview=False):
        """Toggle recording on/off"""
        if not self.is_recording:
            # Start recording
            if not self.init_camera():
                self.print_message("❌ Camera initialization failed")
                return
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = pathlib.Path(VIDEO_DIR) / f"video_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(filepath), fourcc, CAMERA_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT)
            )
            
            if not self.video_writer.isOpened():
                self.print_message("❌ Failed to start recording")
                self.release_camera()
                return
                
            self.is_recording = True
            self.show_preview = with_preview
            self.frame_count = 0
            self.recording_start = time.time()
            
            mode = "with preview" if with_preview else "background"
            self.print_message(f"🔴 Recording: {filepath.name} ({mode})")
            
            if with_preview:
                cv2.namedWindow("Recording", cv2.WINDOW_AUTOSIZE)
        else:
            # Stop recording
            duration = time.time() - self.recording_start if self.recording_start else 0
            self.print_message(f"⏹️  Stopped (duration: {duration:.1f}s, frames: {self.frame_count})")
            
            self.is_recording = False
            self.show_preview = False
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            self.release_camera()
            cv2.destroyAllWindows()
            
    def toggle_preview(self):
        """Toggle preview mode"""
        if self.is_recording:
            self.print_message("⚠️  Cannot change preview during recording")
            return
            
        if not self.show_preview:
            if not self.init_camera():
                self.print_message("❌ Camera initialization failed")
                return
            self.show_preview = True
            cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
            self.print_message("👁️  Preview ON")
        else:
            self.show_preview = False
            self.release_camera()
            cv2.destroyAllWindows()
            self.print_message("👁️  Preview OFF")
            
    def take_screenshot(self):
        """Take a screenshot"""
        need_release = self.camera is None
        
        if not self.init_camera():
            self.print_message("❌ Camera initialization failed")
            return
            
        ret, frame = self.camera.read()
        if ret:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = pathlib.Path(VIDEO_DIR) / f"screenshot_{timestamp}.jpg"
            cv2.imwrite(str(filepath), frame)
            self.print_message(f"📸 Saved: {filepath.name}")
        else:
            self.print_message("❌ Screenshot failed")
            
        if need_release:
            self.release_camera()
            
    def print_message(self, msg):
        """Print a message and return to status line"""
        print(f"\n{msg}")
        time.sleep(0.5)  # Brief pause to see the message
        self.print_status(force=True)  # Force status update
        
    def print_status(self, force=False):
        """Print current status line"""
        # Build status
        if self.is_recording:
            duration = time.time() - self.recording_start
            status = f"🔴 REC {duration:.0f}s"
            if self.show_preview:
                status += " [preview]"
        elif self.show_preview:
            status = "👁️  PREVIEW"
        else:
            status = "⭕ READY"
        
        # Build full status text
        status_text = f"{status} | v:rec w:rec+preview d:preview s:screenshot q:quit"
        
        # Only update if changed or forced or every second for recording timer
        current_time = time.time()
        should_update = (force or 
                        status_text != self.last_status_text or
                        (self.is_recording and current_time - self.last_status_time >= 1.0))
        
        if should_update:
            # Clear and print status line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.write(status_text)
            sys.stdout.flush()
            self.last_status_text = status_text
            self.last_status_time = current_time
        
    def process_frame(self):
        """Process a single camera frame"""
        if not self.camera:
            return
            
        ret, frame = self.camera.read()
        if not ret:
            return
            
        # Write to video if recording
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1
            
        # Show preview if enabled
        if self.show_preview:
            window_name = "Recording" if self.is_recording else "Preview"
            cv2.imshow(window_name, frame)
            
            # Check for window close (ESC key)
            if cv2.waitKey(1) == 27:
                if self.is_recording:
                    self.toggle_recording()
                else:
                    self.toggle_preview()
                    
    def handle_input(self, key):
        """Handle keyboard input"""
        if not key:
            return True
            
        if key == 'q' or key == '\x1b':  # q or ESC
            return False
            
        elif key == 'v':
            self.toggle_recording(with_preview=False)
            
        elif key == 'w':
            self.toggle_recording(with_preview=True)
                
        elif key == 's':
            self.take_screenshot()
            
        elif key == 'd':
            self.toggle_preview()
            
        return True
        
    def run(self):
        """Main run loop"""
        # Clear screen and show header
        os.system('clear' if os.name == 'posix' else 'cls')
        print("USB Camera Video Capture")
        print("=" * 50)
        print("Controls:")
        print("  v  : Start/stop recording (background)")
        print("  w  : Start/stop recording with preview")
        print("  d  : Toggle preview")
        print("  s  : Take screenshot")
        print("  q  : Quit")
        print("=" * 50)
        
        # Test camera
        print(f"\nTesting camera {self.camera_index}...", end='', flush=True)
        test = cv2.VideoCapture(self.camera_index)
        if test.isOpened():
            ret, _ = test.read()
            test.release()
            if ret:
                print(" ✅ OK\n")
            else:
                print(" ❌ Cannot read frames")
                return
        else:
            print(" ❌ Not available")
            return
            
        # Start input handler
        self.input_handler.start()
        
        # Initial status
        self.print_status(force=True)
        
        # Main loop
        running = True
        try:
            while running:
                # Update status (only when needed)
                self.print_status()
                
                # Handle input
                key = self.input_handler.get_key()
                if key:
                    running = self.handle_input(key)
                    self.print_status(force=True)  # Force update after input
                    
                # Process frame
                self.process_frame()
                
                # Small delay
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up all resources"""
        print("\n\nCleaning up...")
        
        # Stop input handler
        self.input_handler.stop()
        
        # Stop recording if active
        if self.is_recording:
            if self.video_writer:
                self.video_writer.release()
                
        # Release camera
        self.release_camera()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("✅ Done")


def find_cameras(max_index=10):
    """Find all available cameras"""
    print("Searching for cameras...")
    cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
                print(f"  ✅ Camera {i}: /dev/video{i}")
            cap.release()
            
    return cameras


def main():
    """Main entry point"""
    import argparse
    
    # Import select for Unix, create dummy for Windows
    try:
        import select
        globals()['select'] = select
    except ImportError:
        # Windows fallback
        class DummySelect:
            @staticmethod
            def select(r, w, x, timeout=0):
                return (r, [], [])
        globals()['select'] = DummySelect()
    
    parser = argparse.ArgumentParser(description='USB Camera Video Capture')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available cameras')
    
    args = parser.parse_args()
    
    if args.list:
        cameras = find_cameras()
        if cameras:
            print(f"\nFound {len(cameras)} camera(s)")
        else:
            print("\nNo cameras found!")
        return
        
    # Run capture
    capture = CameraCapture(camera_index=args.camera)
    capture.run()


if __name__ == "__main__":
    main()
