import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
from Capstone import is_spam_email

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Spam Detection Tool")
        self.root.geometry("800x600")
        
        # Set theme and style
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')  # More modern theme
        except:
            pass  # Fallback to default if theme not available
        
        # Configure colors
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('TLabel', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        self.style.configure('Result.TLabel', font=('Segoe UI', 16, 'bold'))
        
        # Configure progress bar
        self.style.configure("red.Horizontal.TProgressbar", 
                            troughcolor='#f0f0f0', 
                            background='#ff6b6b')
        self.style.configure("green.Horizontal.TProgressbar", 
                            troughcolor='#f0f0f0', 
                            background='#51cf66')
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame, 
            text="Advanced Spam Email Detection", 
            style='Header.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="Email File")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_input_frame = ttk.Frame(file_frame)
        file_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_input_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        browse_button = ttk.Button(
            file_input_frame,
            text="Browse...",
            command=self.browse_file
        )
        browse_button.pack(side=tk.LEFT)
        
        analyze_button = ttk.Button(
            file_frame,
            text="Analyze Email",
            command=self.analyze_email
        )
        analyze_button.pack(pady=(0, 5), anchor=tk.E, padx=5)
        
        # Content preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Email Preview")
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame,
            height=6,
            wrap=tk.WORD
        )
        self.preview_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Result indicator
        result_indicator_frame = ttk.Frame(results_frame)
        result_indicator_frame.pack(fill=tk.X, pady=5)
        
        self.result_label = ttk.Label(
            result_indicator_frame, 
            text="No Analysis Yet", 
            style='Result.TLabel'
        )
        self.result_label.pack(side=tk.LEFT, padx=5)
        
        # Risk score indicator
        score_frame = ttk.Frame(results_frame)
        score_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(score_frame, text="Risk Score:").pack(side=tk.LEFT)
        
        self.progress_frame = ttk.Frame(score_frame)
        self.progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress = ttk.Progressbar(
            self.progress_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            style="red.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=tk.X)
        
        self.score_value = ttk.Label(score_frame, text="0")
        self.score_value.pack(side=tk.LEFT)
        
        # Detailed results with tabs
        self.details_notebook = ttk.Notebook(results_frame)
        self.details_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(self.summary_frame, text='Summary')
        
        self.summary_text = scrolledtext.ScrolledText(
            self.summary_frame,
            wrap=tk.WORD
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Details tab
        self.details_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(self.details_frame, text='Technical Details')
        
        self.details_text = scrolledtext.ScrolledText(
            self.details_frame,
            wrap=tk.WORD
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Raw JSON tab
        self.raw_frame = ttk.Frame(self.details_notebook)
        self.details_notebook.add(self.raw_frame, text='Raw Data')
        
        self.raw_text = scrolledtext.ScrolledText(
            self.raw_frame,
            wrap=tk.WORD,
            font=('Courier', 10)
        )
        self.raw_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Example buttons
        example_frame = ttk.Frame(main_frame)
        example_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(example_frame, text="Try examples:").pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            example_frame,
            text="Spam Email",
            command=lambda: self.load_example("email.txt")
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            example_frame,
            text="Safe Email",
            command=lambda: self.load_example("emailsafe.txt")
        ).pack(side=tk.LEFT, padx=2)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Email File",
            filetypes=(("Email files", "*.eml *.txt"), ("All files", "*.*"))
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.load_email_preview(file_path)
    
    def load_email_preview(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)  # Read first 2000 chars as preview
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, content)
            if len(content) >= 2000:
                self.preview_text.insert(tk.END, "\n[...] (content truncated)")
        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, f"Error loading preview: {str(e)}")
    
    def load_example(self, example_file):
        # Check if the file exists in the current directory
        if os.path.exists(example_file):
            self.file_path_var.set(example_file)
            self.load_email_preview(example_file)
        else:
            messagebox.showinfo("Example", f"Example file '{example_file}' not found in current directory.")
    
    def analyze_email(self):
        file_path = self.file_path_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("Warning", "Please select a valid email file.")
            return
        
        # Update UI
        self.status_var.set("Analyzing email...")
        self.result_label.config(text="Analysis in progress...")
        self.root.update_idletasks()
        
        # Clear previous results
        self.summary_text.delete(1.0, tk.END)
        self.details_text.delete(1.0, tk.END)
        self.raw_text.delete(1.0, tk.END)
        
        # Run analysis in a separate thread to keep UI responsive
        threading.Thread(target=self._run_analysis, args=(file_path,), daemon=True).start()
    
    def _run_analysis(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            
            is_spam, details = is_spam_email(email_content)
            
            # Update UI from main thread
            self.root.after(0, lambda: self._update_results(is_spam, details))
        except Exception as e:
            # Capture the error message to pass to the lambda
            error_message = str(e)
            self.root.after(0, lambda: self._show_error(error_message))
    
    def _update_results(self, is_spam, details):
        # Update result indicator
        if is_spam:
            self.result_label.config(
                text="SPAM DETECTED!", 
                foreground="#e03131"
            )
            self.progress.configure(style="red.Horizontal.TProgressbar")
        else:
            self.result_label.config(
                text="Email appears safe", 
                foreground="#2b8a3e"
            )
            self.progress.configure(style="green.Horizontal.TProgressbar")
        
        # Update score
        total_score = details.get('total_score', 0)
        max_score = 10  # Maximum possible score
        percentage = min(100, int((total_score / max_score) * 100))
        
        self.progress['value'] = percentage
        self.score_value.config(text=f"{total_score}")
        
        # Update summary tab
        self._update_summary(is_spam, details)
        
        # Update details tab
        self._update_details(details)
        
        # Update raw tab
        try:
            self.raw_text.delete(1.0, tk.END)
            self.raw_text.insert(tk.END, json.dumps(details, indent=2))
        except Exception as e:
            self.raw_text.insert(tk.END, f"Error displaying raw data: {str(e)}")
        
        # Update status bar
        self.status_var.set("Analysis completed")
    
    def _update_summary(self, is_spam, details):
        self.summary_text.delete(1.0, tk.END)
        
        # Main verdict
        if is_spam:
            self.summary_text.insert(tk.END, "?? SPAM DETECTED\n\n", "heading")
            self.summary_text.tag_configure("heading", foreground="#e03131", font=("Segoe UI", 12, "bold"))
        else:
            self.summary_text.insert(tk.END, "? Email appears safe\n\n", "heading")
            self.summary_text.tag_configure("heading", foreground="#2b8a3e", font=("Segoe UI", 12, "bold"))
        
        # Summary of findings
        self.summary_text.insert(tk.END, "Summary of findings:\n", "subheading")
        self.summary_text.tag_configure("subheading", font=("Segoe UI", 10, "bold"))
        
        explanations = details.get('explanations', {})
        if explanations:
            for category, explanation in explanations.items():
                if isinstance(explanation, list):
                    self.summary_text.insert(tk.END, f"\n� {category.capitalize()}:\n")
                    for item in explanation:
                        self.summary_text.insert(tk.END, f"  - {item}\n")
                else:
                    self.summary_text.insert(tk.END, f"\n� {explanation}\n")
        else:
            self.summary_text.insert(tk.END, "\nNo specific issues detected.\n")
        
        # Add ML score if available
        if 'ml_spam_probability' in details:
            prob = details['ml_spam_probability']
            self.summary_text.insert(tk.END, f"\nML model confidence: {prob:.1%} likelihood of being spam\n")
        
        # Add recommendations
        self.summary_text.insert(tk.END, "\nRecommendations:\n", "subheading")
        if is_spam:
            self.summary_text.insert(tk.END, "? Do not respond to this email\n")
            self.summary_text.insert(tk.END, "? Do not click any links or open attachments\n")
            self.summary_text.insert(tk.END, "? Consider blocking the sender\n")
            self.summary_text.insert(tk.END, "? Report as spam to your email provider\n")
        else:
            self.summary_text.insert(tk.END, "? This email appears to be legitimate\n")
            if total_score := details.get('total_score', 0) > 0:
                self.summary_text.insert(tk.END, "?? Exercise caution - some minor concerns were detected\n")
    
    def _update_details(self, details):
        self.details_text.delete(1.0, tk.END)
        
        # Subject and sender
        if 'subject' in details:
            self.details_text.insert(tk.END, f"Subject: {details['subject']}\n\n")
        
        if 'sender_ip' in details:
            self.details_text.insert(tk.END, f"Sender IP: {details['sender_ip']}\n")
            ip_score = details.get('scores', {}).get('sender_ip', 0)
            self.details_text.insert(tk.END, f"IP Reputation Score: {ip_score}\n\n")
        
        # Scores section
        self.details_text.insert(tk.END, "Detailed Scores:\n", "heading")
        self.details_text.tag_configure("heading", font=("Segoe UI", 10, "bold"))
        
        scores = details.get('scores', {})
        
        # Subject score
        if 'subject_keywords' in scores:
            self.details_text.insert(tk.END, f"Subject Analysis: {scores['subject_keywords']}\n")
        
        # Body scores
        if 'body_keywords' in scores:
            self.details_text.insert(tk.END, f"Body Keyword Analysis: {scores['body_keywords']}\n")
        
        if 'ml_classification' in scores:
            self.details_text.insert(tk.END, f"ML Classification Score: {scores['ml_classification']}\n")
            if 'ml_spam_probability' in details:
                self.details_text.insert(tk.END, f"ML Spam Probability: {details['ml_spam_probability']:.2%}\n")
        
        # URL scores
        if 'urls' in scores and scores['urls']:
            self.details_text.insert(tk.END, "\nURL Analysis:\n")
            for url, score in scores['urls']:
                self.details_text.insert(tk.END, f"- {url}: Score {score}\n")
        
        # Attachment scores
        if 'attachments' in scores and scores['attachments']:
            self.details_text.insert(tk.END, "\nAttachment Analysis:\n")
            for name, score in scores['attachments']:
                self.details_text.insert(tk.END, f"- {name}: Score {score}\n")
        
        # Total score
        self.details_text.insert(tk.END, f"\nTotal Risk Score: {details.get('total_score', 0)}\n")
        self.details_text.insert(tk.END, f"Verdict: {'SPAM' if details.get('total_score', 0) >= 3 else 'NOT SPAM'}\n")
    
    def _show_error(self, error_message):
        self.status_var.set(f"Error: {error_message}")
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_message}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()

root = tk.TK()
app = SpamDetectorGUI(root)
root.mainloop()