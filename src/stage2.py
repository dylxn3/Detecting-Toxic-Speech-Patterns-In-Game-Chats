# stage2.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch
import ast
import time
from tqdm import tqdm
import gc
import warnings
from datetime import datetime, timedelta
import psutil
import humanize
warnings.filterwarnings('ignore')

# Try to install required packages if not present
try:
    import humanize
except ImportError:
    os.system('pip install humanize')
    import humanize

try:
    import psutil
except ImportError:
    os.system('pip install psutil')
    import psutil

class ProgressTracker:
    """Tracks progress and provides status updates"""
    
    def __init__(self, total_steps, description="Overall Progress"):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.step_times = []
        self.step_names = []
        
    def update(self, step_name=None):
        """Update progress and show status"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if step_name:
            self.step_names.append(step_name)
            self.step_times.append(elapsed)
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_time_per_step * remaining_steps
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            
            sys.stdout.write('\033[K')
            print(f"\r{'='*60}")
            print(f"{self.description}")
            print(f"{'='*60}")
            print(f"Elapsed: {humanize.naturaldelta(timedelta(seconds=int(elapsed)))}")
            print(f"Progress: {self.current_step}/{self.total_steps} steps ({self.current_step/self.total_steps*100:.1f}%)")
            print(f"ETA: {eta.strftime('%H:%M:%S')} ({humanize.naturaldelta(timedelta(seconds=int(eta_seconds)))})")
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            print(f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% ({humanize.naturalsize(memory.used)}/{humanize.naturalsize(memory.total)})")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {gpu_memory:.1f}GB/{gpu_total:.1f}GB ({gpu_memory/gpu_total*100:.1f}%)")
            
            if step_name:
                print(f"Current step: {step_name}")
            
            if len(self.step_times) > 1:
                print(f"\nRecent steps:")
                for i in range(max(0, len(self.step_times)-3), len(self.step_times)):
                    step_time = self.step_times[i] - (self.step_times[i-1] if i > 0 else 0)
                    print(f"  \u2022 {self.step_names[i]}: {step_time:.1f}s")
            
            sys.stdout.flush()
    
    def finish(self):
        """Show final statistics"""
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"{self.description} COMPLETED!")
        print(f"{'='*60}")
        print(f"Total time: {humanize.naturaldelta(timedelta(seconds=int(total_time)))}")
        print(f"Average step time: {total_time/self.total_steps:.1f}s")
        
        if self.step_times:
            print(f"\nStep breakdown:")
            prev_time = 0
            for i, (name, time_point) in enumerate(zip(self.step_names, self.step_times)):
                step_duration = time_point - prev_time
                print(f"  {i+1}. {name}: {step_duration:.1f}s")
                prev_time = time_point
        
        print(f"{'='*60}\n")

class TimeLogger:
    """Logs time for specific operations"""
    
    def __init__(self):
        self.logs = []
        self.current_start = None
        self.current_name = None
    
    def start(self, name):
        """Start timing an operation"""
        self.current_start = time.time()
        self.current_name = name
        print(f"\nStarting: {name}")
        sys.stdout.flush()
    
    def stop(self):
        """Stop timing current operation"""
        if self.current_start and self.current_name:
            duration = time.time() - self.current_start
            self.logs.append({
                'operation': self.current_name,
                'duration': duration,
                'timestamp': datetime.now()
            })
            print(f"Completed: {self.current_name} ({duration:.2f}s)")
            self.current_start = None
            self.current_name = None
            return duration
    
    def summary(self):
        """Print summary of all timed operations"""
        if not self.logs:
            return
        
        print(f"\n{'='*60}")
        print("TIME SUMMARY")
        print(f"{'='*60}")
        
        total_time = sum(log['duration'] for log in self.logs)
        sorted_logs = sorted(self.logs, key=lambda x: x['duration'], reverse=True)
        
        for log in sorted_logs:
            percentage = (log['duration'] / total_time) * 100
            print(f"  \u2022 {log['operation']}: {log['duration']:.2f}s ({percentage:.1f}%)")
        
        print(f"{'='*60}")
        print(f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*60}\n")

def detect_sarcasm_batch(texts, sarcasm_classifier, batch_size=64, time_logger=None):
    """Process sarcasm detection in batches for efficiency"""
    if sarcasm_classifier is None:
        return [False] * len(texts)
    
    if time_logger:
        time_logger.start("Sarcasm Detection")
    
    results = []
    
    try:
        pbar = tqdm(total=len(texts), desc="Detecting sarcasm", unit="text", 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            pbar.set_description(f"Detecting sarcasm (Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1})")
            
            try:
                batch_predictions = sarcasm_classifier(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    batch_size=batch_size
                )
                
                for pred in batch_predictions:
                    if isinstance(pred, list):
                        label = pred[0]['label'].lower()
                    else:
                        label = pred['label'].lower()
                    
                    is_sarcastic = any(indicator in label for indicator in 
                                      ['sarcasm', 'sarcastic', 'label_1', '1', 'positive'])
                    results.append(is_sarcastic)
            except:
                for text in batch_texts:
                    result = sarcasm_classifier(text, truncation=True, max_length=512)[0]
                    label = result['label'].lower()
                    is_sarcastic = any(indicator in label for indicator in 
                                      ['sarcasm', 'sarcastic', 'label_1', '1', 'positive'])
                    results.append(is_sarcastic)
            
            pbar.update(len(batch_texts))
            
            if i % (batch_size * 10) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        pbar.close()
                    
    except Exception as e:
        print(f"\nError in batch sarcasm detection: {e}")
        remaining = len(texts) - len(results)
        for i in tqdm(range(remaining), desc="Fallback processing"):
            try:
                result = sarcasm_classifier(texts[len(results)], truncation=True, max_length=512)[0]
                label = result['label'].lower()
                is_sarcastic = any(indicator in label for indicator in 
                                  ['sarcasm', 'sarcastic', 'label_1', '1', 'positive'])
                results.append(is_sarcastic)
            except:
                results.append(False)
    
    if time_logger:
        time_logger.stop()
    
    return results

def predict_bert_batch(texts, bert_model, batch_size=32, time_logger=None):
    """Optimized BERT predictions - IGNORE labels, use ONLY scores"""
    if time_logger:
        time_logger.start("BERT Predictions")
    
    preds = []
    confidence_scores = []
    
    pbar = tqdm(total=len(texts), desc="BERT predictions", unit="text",
               bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    
    # Show diagnostic
    print("\n" + "="*60)
    print("BERT MODEL DIAGNOSTIC")
    print("="*60)
    print("\nNOTE: Model outputs 'toxic' for everything - we'll use ONLY the score!")
    print("Score > 0.5 = Toxic, Score < 0.5 = Non-toxic\n")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        pbar.set_description(f"BERT predictions (Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1})")
        
        try:
            results = bert_model(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                batch_size=batch_size
            )
            
            for j, r in enumerate(results):
                # Extract the result
                if isinstance(r, list):
                    result = r[0]
                else:
                    result = r
                
                # Get the score - this is the ONLY thing that matters
                score = result['score']
                
                # ============================================================
                # THE FIX: Ignore the label completely, use only the score
                # ============================================================
                # Score is the toxicity probability
                # If score > 0.5, it's toxic; if score < 0.5, it's non-toxic
                
                TOXICITY_THRESHOLD = 0.5
                
                if score > TOXICITY_THRESHOLD:
                    pred_label = 1
                    confidence = score
                else:
                    pred_label = 0
                    confidence = 1 - score  # Confidence in non-toxic prediction
                
                preds.append(pred_label)
                confidence_scores.append(confidence)
                
                # Print first few to verify
                if i == 0 and j < 5:
                    print(f"\nSample {j}:")
                    print(f"  Text: {batch[j][:50]}...")
                    print(f"  Score: {score:.4f} -> {'TOXIC' if pred_label == 1 else 'NON-TOXIC'}")
                    
        except Exception as e:
            print(f"\nError in batch {i//batch_size + 1}: {e}")
            # Fallback: assume non-toxic
            for _ in batch:
                preds.append(0)
                confidence_scores.append(0.5)
        
        pbar.update(len(batch))
        
        if i % (batch_size * 20) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    pbar.close()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("BERT PREDICTION SUMMARY")
    print(f"{'='*60}")
    toxic_count = sum(preds)
    print(f"Total predictions: {len(preds)}")
    print(f"Predicted toxic: {toxic_count} ({toxic_count/len(preds)*100:.2f}%)")
    print(f"Predicted non-toxic: {len(preds)-toxic_count} ({(len(preds)-toxic_count)/len(preds)*100:.2f}%)")
    
    # This should now match your actual data distribution (~5-10% toxic)
    if toxic_count/len(preds) > 20:
        print("\n⚠️  WARNING: Still predicting too many toxic comments!")
        print("Try increasing the threshold to 0.7 or 0.8")
    elif toxic_count/len(preds) < 1:
        print("\n⚠️  WARNING: Predicting too few toxic comments!")
        print("Try decreasing the threshold to 0.3 or 0.4")
    else:
        print("\n✅ BERT predictions look reasonable!")
    
    if time_logger:
        time_logger.stop()
    
    return np.array(preds)

def main():
    time_logger = TimeLogger()
    overall_start = time.time()
    
    print("\n" + "="*60)
    print("STAGE 2: TOXICITY DETECTION MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Load Stage 1 output
    input_file = "../output/stage1_outputs/stage1_output.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    print(f"Loading data from {input_file}...")
    time_logger.start("Data Loading")
    
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)
    
    print(f"Loaded {len(df):,} comments")
    
    # Verify labels are correct
    print(f"\nLabel distribution:")
    print(f"  Toxic (1): {df['toxic_label'].sum():,} ({df['toxic_label'].mean()*100:.2f}%)")
    print(f"  Non-toxic (0): {len(df) - df['toxic_label'].sum():,} ({(1-df['toxic_label'].mean())*100:.2f}%)")
    
    time_logger.stop()
    
    # Parse failure tags
    time_logger.start("Parsing Failure Tags")
    print("\nParsing failure tags...")
    df['failure_tags'] = df['failure_tags'].apply(ast.literal_eval)
    time_logger.stop()

    # Show initial tag distribution
    all_tags = []
    for tags in df['failure_tags']:
        all_tags.extend(tags)
    
    from collections import Counter
    tag_counts = Counter(all_tags)
    print("\nInitial failure tags from Stage 1:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count} ({count/len(df)*100:.2f}%)")

    # Adding sarcasm to failure tags
    use_sarcasm = True
    
    if use_sarcasm:
        print("\n" + "-"*60)
        print("SARCASTIC DETECTION PHASE")
        print("-"*60 + "\n")
        
        try:
            print("Attempting to load Hugging Face sarcasm detection model...")
            time_logger.start("Load Sarcasm Model")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU available: {gpu_memory:.2f} GB")
            
            sarcasm_classifier = pipeline(
                "text-classification",
                model="TomBearss/sarcasm-detection",
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=64
            )
            print("Model loaded successfully!")
            time_logger.stop()
            
            sarcasm_detected = detect_sarcasm_batch(
                df['cleaned_text'].tolist(), 
                sarcasm_classifier, 
                batch_size=64,
                time_logger=time_logger
            )
            
        except Exception as e:
            print(f"Hugging Face model failed: {e}")
            print("Switching to lightweight rule-based detection...")
            
            time_logger.start("Rule-based Sarcasm Detection")
            sarcasm_detected = []
            pbar = tqdm(total=len(df), desc="Rule-based detection", unit="text")
            
            sarcasm_phrases = [
                "yeah right", "oh great", "just what i needed", "thanks a lot",
                "very funny", "how original", "big surprise", "as if"
            ]
            
            for text in df['cleaned_text']:
                text_lower = text.lower()
                is_sarcastic = any(phrase in text_lower for phrase in sarcasm_phrases)
                if not is_sarcastic and ('!!!' in text or '???' in text):
                    is_sarcastic = True
                sarcasm_detected.append(is_sarcastic)
                pbar.update(1)
            
            pbar.close()
            time_logger.stop()
        
        time_logger.start("Adding Sarcasm Tags")
        print("\nAdding sarcasm tags to failure tags...")
        sarcasm_count = 0
        for idx, is_sarcastic in enumerate(tqdm(sarcasm_detected, desc="Updating tags")):
            if is_sarcastic:
                df.at[idx, 'failure_tags'] = df.at[idx, 'failure_tags'] + ["sarcasm"]
                sarcasm_count += 1
        
        print(f"\nSarcasm detected in {sarcasm_count:,}/{len(df):,} texts ({sarcasm_count/len(df)*100:.2f}%)")
        time_logger.stop()
        
        del sarcasm_detected
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Split data
    print("\n" + "-"*60)
    print("DATA SPLITTING")
    print("-"*60 + "\n")
    
    time_logger.start("Data Splitting")
    
    X = df['cleaned_text']
    y = df['toxic_label']

    X_train, X_test, y_train, y_test, train_tags, test_tags = train_test_split(
        X, y, df['failure_tags'], test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    print(f"Toxic ratio in training: {y_train.mean():.3f}")
    print(f"Toxic ratio in test: {y_test.mean():.3f}")
    
    time_logger.stop()

    # Model 1 - TF-IDF
    print("\n" + "-"*60)
    print("TRAINING TF-IDF + LOGISTIC REGRESSION")
    print("-"*60 + "\n")
    
    time_logger.start("TF-IDF Vectorization")
    
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32
    )
    
    print("Fitting TF-IDF vectorizer...")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    time_logger.stop()

    print("\nTraining Logistic Regression model...")
    time_logger.start("Logistic Regression Training")
    
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        solver='saga',
        tol=0.01
    )
    
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    
    print("TF-IDF model training complete!")
    time_logger.stop()

    # Model 2 - BERT
    print("\n" + "-"*60)
    print("LOADING BERT TOXICITY MODEL")
    print("-"*60 + "\n")
    
    time_logger.start("Load BERT Model")
    
    bert_model = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
        batch_size=64
    )
    
    print("BERT model loaded!")
    time_logger.stop()
    
    print("\nMaking BERT predictions...")
    y_pred_bert = predict_bert_batch(list(X_test), bert_model, batch_size=32, time_logger=time_logger)

    # Ensure all arrays have the same length
    min_len = min(len(y_test), len(y_pred_bert), len(y_pred_lr))
    if len(y_test) != min_len or len(y_pred_bert) != min_len or len(y_pred_lr) != min_len:
        print(f"\nTruncating arrays to common length: {min_len}")
        y_test = y_test[:min_len]
        y_pred_bert = y_pred_bert[:min_len]
        y_pred_lr = y_pred_lr[:min_len]
        test_tags = test_tags[:min_len]
    
    # ============================================================
    # DIAGNOSTIC OUTPUT
    # ============================================================
    print("\n" + "="*60)
    print("PREDICTION DIAGNOSTIC")
    print("="*60)

    print(f"\nPREDICTION DISTRIBUTION:")
    print(f"True toxic rate: {y_test.mean():.4f} ({y_test.sum()}/{len(y_test)})")
    print(f"BERT predicted toxic: {y_pred_bert.mean():.4f} ({y_pred_bert.sum()}/{len(y_pred_bert)})")
    print(f"TF-IDF predicted toxic: {y_pred_lr.mean():.4f} ({y_pred_lr.sum()}/{len(y_pred_lr)})")

    print(f"\nSAMPLE PREDICTIONS (first 20):")
    print(f"{'Index':<6} {'True':<6} {'BERT':<6} {'TF-IDF':<6} Text Snippet")
    print("-"*70)
    for i in range(min(20, len(y_test))):
        true_val = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        print(f"{i:<6} {true_val:<6} {y_pred_bert[i]:<6} {y_pred_lr[i]:<6} {str(X_test.iloc[i])[:50] if hasattr(X_test, 'iloc') else str(X_test[i])[:50]}...")

    print(f"\nBERT Prediction Summary:")
    print(f"  \u2022 Predicting toxic: {y_pred_bert.sum()} comments")
    print(f"  \u2022 Predicting non-toxic: {len(y_pred_bert) - y_pred_bert.sum()} comments")
    print(f"  \u2022 Toxic ratio in predictions: {y_pred_bert.mean()*100:.2f}%")
    print(f"  \u2022 Actual toxic ratio: {y_test.mean()*100:.2f}%")

    if y_pred_bert.mean() > 0.8:
        print("\nWARNING: BERT is predicting >80% as toxic - this is wrong!")
        print("Check the diagnostic output above - the model may be returning unexpected labels.")
    elif y_pred_bert.mean() < 0.05:
        print("\nWARNING: BERT is predicting <5% as toxic - this is wrong!")
    
    # Compute metrics
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    # Overall metrics
    print("\n" + "-"*60)
    print("OVERALL METRICS")
    print("-"*60)
    
    time_logger.start("Computing Metrics")
    
    overall_metrics = {
        "TF-IDF": compute_metrics(y_test, y_pred_lr),
        "BERT": compute_metrics(y_test, y_pred_bert)
    }

    overall_df = pd.DataFrame(overall_metrics).round(3)
    print("\n" + overall_df.to_string())
    
    # Metrics per failure phenomenon
    phenomena = ["slang", "repeated_letters", "all_caps", "excessive_punctuation"]
    if use_sarcasm:
        phenomena.append("sarcasm")

    per_tag_metrics = {}
    for tag in phenomena:
        indices = np.array([tag in tags for tags in test_tags])
        if indices.sum() == 0:
            print(f"\nNo examples found with tag: {tag}")
            continue
        
        print(f"\nAnalyzing {indices.sum():,} examples with tag: {tag}")
        per_tag_metrics[tag] = {
            "TF-IDF": compute_metrics(y_test[indices], y_pred_lr[indices]),
            "BERT": compute_metrics(y_test[indices], y_pred_bert[indices])
        }

    print("\n" + "-"*60)
    print("METRICS PER FAILURE PHENOMENON")
    print("-"*60)
    
    for tag, metrics in per_tag_metrics.items():
        print(f"\nTag: {tag.upper()}")
        print(pd.DataFrame(metrics).round(3))
    
    time_logger.stop()

    # Create output directory for Stage 2
    output_folder = "../output/stage2_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Save results
    print("\n" + "-"*60)
    print("SAVING RESULTS")
    print("-"*60 + "\n")
    
    time_logger.start("Saving Results")

    df_test = pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "true_label": y_test.reset_index(drop=True),
        "pred_lr": y_pred_lr,
        "pred_bert": y_pred_bert,
        "failure_tags": test_tags.reset_index(drop=True).apply(lambda x: str(x))
    })
    
    df_test['misclassified_lr'] = df_test['true_label'] != df_test['pred_lr']
    df_test['misclassified_bert'] = df_test['true_label'] != df_test['pred_bert']

    # Save predictions
    predictions_file = os.path.join(output_folder, "stage2_predictions.csv.gz")
    df_test.to_csv(predictions_file, index=False, compression='gzip')
    print(f"Predictions saved to: {predictions_file}")
    
    # Save overall metrics
    overall_metrics_df = pd.DataFrame(overall_metrics).round(3)
    overall_metrics_df.to_csv(os.path.join(output_folder, "overall_metrics.csv"))
    
    # Save per-tag metrics
    if per_tag_metrics:
        per_tag_rows = []
        for tag, metrics in per_tag_metrics.items():
            row = {'tag': tag}
            row.update({f"tfidf_{k}": v for k, v in metrics['TF-IDF'].items()})
            row.update({f"bert_{k}": v for k, v in metrics['BERT'].items()})
            per_tag_rows.append(row)
        
        per_tag_df = pd.DataFrame(per_tag_rows)
        per_tag_df.to_csv(os.path.join(output_folder, "per_tag_metrics.csv"), index=False)
    
    # Summary statistics
    summary = {
        'total_samples': len(df_test),
        'toxic_samples': df_test['true_label'].sum(),
        'toxic_ratio': df_test['true_label'].mean(),
        'tfidf_misclassified': df_test['misclassified_lr'].sum(),
        'bert_misclassified': df_test['misclassified_bert'].sum(),
        'tfidf_misclassification_rate': df_test['misclassified_lr'].mean(),
        'bert_misclassification_rate': df_test['misclassified_bert'].mean(),
        'both_correct': ((~df_test['misclassified_lr']) & (~df_test['misclassified_bert'])).sum(),
        'both_wrong': (df_test['misclassified_lr'] & df_test['misclassified_bert']).sum(),
        'tfidf_only_correct': ((~df_test['misclassified_lr']) & df_test['misclassified_bert']).sum(),
        'bert_only_correct': (df_test['misclassified_lr'] & (~df_test['misclassified_bert'])).sum(),
    }
    
    pd.DataFrame([summary]).to_csv(os.path.join(output_folder, "stage2_summary.csv"), index=False)
    time_logger.stop()

    # Generate graphs
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60 + "\n")
    
    time_logger.start("Generating Graphs")
    
    # Sample data for plotting
    plot_sample_size = min(10000, len(df_test))
    df_plot = df_test.sample(n=plot_sample_size, random_state=42) if len(df_test) > 10000 else df_test
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 1. Overall model comparison
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame({
        "TF-IDF": [overall_metrics["TF-IDF"]["f1"], 
                   overall_metrics["TF-IDF"]["precision"], 
                   overall_metrics["TF-IDF"]["recall"]],
        "BERT": [overall_metrics["BERT"]["f1"], 
                 overall_metrics["BERT"]["precision"], 
                 overall_metrics["BERT"]["recall"]]
    }, index=["F1 Score", "Precision", "Recall"])

    ax = metrics_df.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("Overall Model Performance Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(fontsize=11)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "model_overall_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-failure phenomenon F1 comparison
    if per_tag_metrics:
        plt.figure(figsize=(12, 6))
        per_tag_f1 = pd.DataFrame({
            tag: {
                "TF-IDF": per_tag_metrics[tag]["TF-IDF"]["f1"], 
                "BERT": per_tag_metrics[tag]["BERT"]["f1"]
            }
            for tag in per_tag_metrics
        }).T

        ax = per_tag_f1.plot(kind="bar", figsize=(12, 6), width=0.8)
        plt.title("F1 Score per Failure Phenomenon", fontsize=14, fontweight='bold')
        plt.ylabel("F1 Score", fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.legend(fontsize=11)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "f1_per_failure_tag.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Non-Toxic', 'Toxic'],
                yticklabels=['Non-Toxic', 'Toxic'])
    axes[0].set_title('TF-IDF Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    cm_bert = confusion_matrix(y_test, y_pred_bert)
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Non-Toxic', 'Toxic'],
                yticklabels=['Non-Toxic', 'Toxic'])
    axes[1].set_title('BERT Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    time_logger.stop()

    # Save a summary report
    with open(os.path.join(output_folder, "summary_report.txt"), 'w') as f:
        f.write("STAGE 2 PROCESSING SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total test samples: {len(df_test):,}\n")
        f.write(f"Toxic samples in test: {df_test['true_label'].sum():,} ({df_test['true_label'].mean()*100:.2f}%)\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*30 + "\n")
        for model in ["TF-IDF", "BERT"]:
            f.write(f"\n{model}:\n")
            for metric, value in overall_metrics[model].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\n\nPER-TAG METRICS (F1 SCORES):\n")
        f.write("-"*30 + "\n")
        for tag, metrics in per_tag_metrics.items():
            f.write(f"\n{tag}:\n")
            f.write(f"  TF-IDF F1: {metrics['TF-IDF']['f1']:.4f}\n")
            f.write(f"  BERT F1: {metrics['BERT']['f1']:.4f}\n")
        
        f.write("\n\nMISCLASSIFICATION SUMMARY:\n")
        f.write("-"*30 + "\n")
        f.write(f"TF-IDF misclassifications: {summary['tfidf_misclassified']:,} ({summary['tfidf_misclassification_rate']*100:.2f}%)\n")
        f.write(f"BERT misclassifications: {summary['bert_misclassified']:,} ({summary['bert_misclassification_rate']*100:.2f}%)\n")
        f.write(f"Both models correct: {summary['both_correct']:,}\n")
        f.write(f"Both models wrong: {summary['both_wrong']:,}\n")
        
        f.write("\n\nFILES GENERATED:\n")
        f.write("-"*30 + "\n")
        f.write("stage2_predictions.csv.gz - Test set predictions\n")
        f.write("overall_metrics.csv - Overall model performance\n")
        f.write("per_tag_metrics.csv - Performance by failure phenomenon\n")
        f.write("stage2_summary.csv - Summary statistics\n")
        f.write("model_overall_metrics.png - Bar chart of overall metrics\n")
        f.write("f1_per_failure_tag.png - F1 scores by phenomenon\n")
        f.write("confusion_matrices.png - Confusion matrices for both models\n")
        f.write("summary_report.txt - This summary file\n")

    # Final summary
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("STAGE 2 COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")
    
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Processed {len(df):,} comments")
    print(f"Test set size: {len(X_test):,} comments")
    
    print(f"\nAll outputs saved to: {output_folder}")
    print("\nFiles generated:")
    print(f"  \u2022 stage2_predictions.csv.gz - Test set predictions")
    print(f"  \u2022 overall_metrics.csv - Overall model performance")
    print(f"  \u2022 per_tag_metrics.csv - Performance by failure phenomenon")
    print(f"  \u2022 stage2_summary.csv - Summary statistics")
    print(f"  \u2022 model_overall_metrics.png - Overall metrics chart")
    print(f"  \u2022 f1_per_failure_tag.png - F1 scores by phenomenon")
    print(f"  \u2022 confusion_matrices.png - Confusion matrices")
    print(f"  \u2022 summary_report.txt - Text summary")
    
    # Show time summary
    time_logger.summary()

# Run pipeline
if __name__ == "__main__":
    main()