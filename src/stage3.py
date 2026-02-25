# stage3.py - Complete Linguistic Pattern Analysis for Thesis

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import ast
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class LinguisticAnalyzer:
    """
    Comprehensive analyzer for toxic speech patterns in multiplayer game chat.
    
    This class performs the linguistic analysis promised in Thesis Objectives 3 & 4:
    - Identifies statistical patterns in toxic vs non-toxic messages
    - Analyzes word choice, message structure, and linguistic features
    - Generates visualizations to support interpretation of results
    """
    
    def __init__(self, df):
        """
        Initialize the analyzer with the dataset.
        
        Args:
            df: DataFrame containing comments with 'toxic_label' and text columns
        """
        self.df = df
        self.toxic_df = df[df['toxic_label'] == 1]
        self.nontoxic_df = df[df['toxic_label'] == 0]
        
        print(f"\n ")
        print("LINGUISTIC ANALYZER INITIALIZED")
        print(f"Total comments: {len(df):,}")
        print(f"Toxic comments: {len(self.toxic_df):,} ({len(self.toxic_df)/len(df)*100:.2f}%)")
        print(f"Non-toxic comments: {len(self.nontoxic_df):,} ({len(self.nontoxic_df)/len(df)*100:.2f}%)")
        
    def analyze_message_length(self):
        """
        Objective 3: Statistical pattern analysis - message length
        
        Compares message length between toxic and non-toxic comments.
        Longer messages might indicate more emotional investment or detailed insults.
        Shorter messages might be quick slurs or spam.
        """
        print("\n"  )
        print("ANALYSIS 1: MESSAGE LENGTH PATTERNS")
        
        # Determine which text column to use
        text_column = 'original_text' if 'original_text' in self.df.columns else 'cleaned_text'
        
        # Create word_count and char_count columns
        self.df['word_count'] = self.df[text_column].astype(str).str.split().str.len()
        self.df['char_count'] = self.df[text_column].astype(str).str.len()
        
        # Update toxic and non-toxic dataframes with new columns
        self.toxic_df = self.df[self.df['toxic_label'] == 1]
        self.nontoxic_df = self.df[self.df['toxic_label'] == 0]
        
        toxic_word_avg = self.toxic_df['word_count'].mean()
        nontoxic_word_avg = self.nontoxic_df['word_count'].mean()
        toxic_char_avg = self.toxic_df['char_count'].mean()
        nontoxic_char_avg = self.nontoxic_df['char_count'].mean()
        
        print(f"\nWORD COUNT ANALYSIS:")
        print(f"  Toxic comments: {toxic_word_avg:.2f} words on average")
        print(f"  Non-toxic comments: {nontoxic_word_avg:.2f} words on average")
        print(f"  Difference: {toxic_word_avg - nontoxic_word_avg:.2f} words")
        print(f"  Ratio (Toxic/Non-toxic): {toxic_word_avg/nontoxic_word_avg:.2f}x")
        
        print(f"\nCHARACTER COUNT ANALYSIS:")
        print(f"  Toxic comments: {toxic_char_avg:.2f} characters on average")
        print(f"  Non-toxic comments: {nontoxic_char_avg:.2f} characters on average")
        print(f"  Difference: {toxic_char_avg - nontoxic_char_avg:.2f} characters")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            self.toxic_df['word_count'].dropna(),
            self.nontoxic_df['word_count'].dropna()
        )
        
        print(f"\nSTATISTICAL SIGNIFICANCE (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✓ Result is statistically significant (p < 0.05)")
            print(f"    This means the length difference is unlikely to be due to chance")
        else:
            print(f"  ✗ Result is NOT statistically significant")
        
        return {
            'toxic_word_avg': toxic_word_avg,
            'nontoxic_word_avg': nontoxic_word_avg,
            'toxic_char_avg': toxic_char_avg,
            'nontoxic_char_avg': nontoxic_char_avg,
            't_stat': t_stat,
            'p_value': p_value
        }
    
    def analyze_vocabulary_diversity(self):
        """
        Objective 3: Statistical pattern analysis - vocabulary richness
        
        Measures vocabulary diversity using Type-Token Ratio (TTR).
        Higher TTR indicates more diverse vocabulary.
        Lower TTR might indicate repetitive insults or limited vocabulary.
        """
        print("\n"  )
        print("ANALYSIS 2: VOCABULARY DIVERSITY")
        
        def type_token_ratio(texts):
            """
            Calculate Type-Token Ratio (TTR)
            
            TTR = (number of unique words) / (total number of words)
            - Higher TTR (closer to 1): More diverse vocabulary
            - Lower TTR (closer to 0): More repetitive vocabulary
            """
            all_words = []
            for text in texts.dropna():
                all_words.extend(str(text).split())
            
            if len(all_words) == 0:
                return 0
            
            unique_words = set(all_words)
            ttr = len(unique_words) / len(all_words)
            return ttr
        
        # Use cleaned_text for vocabulary analysis to avoid punctuation artifacts
        text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'original_text'
        
        toxic_ttr = type_token_ratio(self.toxic_df[text_column])
        nontoxic_ttr = type_token_ratio(self.nontoxic_df[text_column])
        
        print(f"\nTYPE-TOKEN RATIO (higher = more diverse vocabulary):")
        print(f"  Toxic vocabulary diversity: {toxic_ttr:.4f}")
        print(f"  Non-toxic vocabulary diversity: {nontoxic_ttr:.4f}")
        print(f"  Difference: {toxic_ttr - nontoxic_ttr:.4f}")
        
        if toxic_ttr > nontoxic_ttr:
            print(f"  → Toxic comments use MORE diverse vocabulary")
        else:
            print(f"  → Toxic comments use MORE REPETITIVE vocabulary")
        
        return {
            'toxic_ttr': toxic_ttr,
            'nontoxic_ttr': nontoxic_ttr
        }
    
    def analyze_profanity_patterns(self):
        """
        Objective 3: Statistical pattern analysis - profanity usage
        
        Analyzes how profanity is used in toxic vs non-toxic comments.
        This helps understand if profanity alone indicates toxicity or if context matters.
        """
        print("\n"  )
        print("ANALYSIS 3: PROFANITY PATTERNS")
        
        # Comprehensive profanity patterns (expand based on your data)
        profanity_patterns = {
            'fuck': r'\b(fuck|fck|fk|f\*ck|fuk|f\*\*\*)\w*\b',
            'shit': r'\b(sh[i1]t|sh\*t|sht|sh\*\*)\w*\b',
            'bitch': r'\b(b[i1]tch|b\*tch|btch|b\*\*\*h)\w*\b',
            'ass': r'\b(ass|a\*{2}|as$|a\*\*)\w*\b',
            'dick': r'\b(d[i1]ck|d\*ck|d\*\*k)\w*\b',
            'cock': r'\b(cock|c\*ck|c\*\*k)\w*\b',
            'piss': r'\b(p[i1]ss|p\*ss|p\*\*)\w*\b',
            'cunt': r'\b(cunt|c\*nt|c\*\*t)\w*\b',
            'twat': r'\b(twat|tw\*t)\w*\b',
            'slag': r'\b(slag)\w*\b'
        }
        
        def count_profanity_by_type(text):
            """Count each type of profanity in a text"""
            if pd.isna(text) or text == '':
                return {category: 0 for category in profanity_patterns}
            
            counts = {}
            text_lower = str(text).lower()
            for category, pattern in profanity_patterns.items():
                counts[category] = len(re.findall(pattern, text_lower))
            return counts
        
        # Apply profanity counting using original_text to catch variations
        text_column = 'original_text' if 'original_text' in self.df.columns else 'cleaned_text'
        
        print("Analyzing profanity patterns...")
        profanity_counts = self.df[text_column].apply(count_profanity_by_type)
        
        # Convert to DataFrame
        profanity_df = pd.DataFrame(profanity_counts.tolist())
        self.df = pd.concat([self.df, profanity_df], axis=1)
        
        # Update toxic and non-toxic dataframes
        self.toxic_df = self.df[self.df['toxic_label'] == 1]
        self.nontoxic_df = self.df[self.df['toxic_label'] == 0]
        
        # Calculate statistics
        print(f"\nPROFANITY USAGE STATISTICS:")
        print(f"{'Category':<15} {'Toxic %':<12} {'Non-toxic %':<15} {'Ratio':<10}")
        print("-" * 55)
        
        results = {}
        for category in profanity_patterns.keys():
            toxic_pct = (self.toxic_df[category] > 0).sum() / len(self.toxic_df) * 100
            nontoxic_pct = (self.nontoxic_df[category] > 0).sum() / len(self.nontoxic_df) * 100
            ratio = toxic_pct / nontoxic_pct if nontoxic_pct > 0 else float('inf')
            
            results[category] = {
                'toxic_pct': toxic_pct,
                'nontoxic_pct': nontoxic_pct,
                'ratio': ratio
            }
            
            print(f"{category:<15} {toxic_pct:.2f}%      {nontoxic_pct:.2f}%        {ratio:.2f}x")
        
        # Overall profanity presence
        self.df['has_profanity'] = (self.df[list(profanity_patterns.keys())] > 0).any(axis=1)
        self.toxic_df = self.df[self.df['toxic_label'] == 1]
        self.nontoxic_df = self.df[self.df['toxic_label'] == 0]
        
        toxic_profanity_pct = self.toxic_df['has_profanity'].sum() / len(self.toxic_df) * 100
        nontoxic_profanity_pct = self.nontoxic_df['has_profanity'].sum() / len(self.nontoxic_df) * 100
        
        print(f"\nOVERALL PROFANITY PRESENCE:")
        print(f"  Toxic comments with profanity: {toxic_profanity_pct:.2f}%")
        print(f"  Non-toxic comments with profanity: {nontoxic_profanity_pct:.2f}%")
        print(f"  Toxic comments are {toxic_profanity_pct/nontoxic_profanity_pct:.2f}x more likely to contain profanity")
        
        return results
    
    def analyze_pronoun_usage(self, silent=False):
        """
        Objective 3: Statistical pattern analysis - pronoun patterns
        
        Analyzes pronoun usage to understand targeting patterns:
        - First person (I, me): Self-focused
        - Second person (you): Directly targeting others
        - Third person (they): Talking about others
        """
        if not silent:
            print("\n"  )
            print("ANALYSIS 4: PRONOUN USAGE PATTERNS")
        
        pronoun_patterns = {
            'first_person_singular': r'\b(i|me|my|mine|myself)\b',
            'first_person_plural': r'\b(we|us|our|ours|ourselves)\b',
            'second_person': r'\b(you|your|yours|yourself|yourselves|u|ur)\b',
            'third_person': r'\b(he|him|his|she|her|they|them|their|theirs)\b'
        }
        
        # Use original_text for pronoun analysis
        text_column = 'original_text' if 'original_text' in self.df.columns else 'cleaned_text'
        
        if not silent:
            print(f"\nPRONOUN USAGE (% of messages containing each type):")
            print(f"{'Pronoun Type':<25} {'Toxic %':<12} {'Non-toxic %':<15} {'Ratio':<10}")
            print("-" * 65)
        
        results = {}
        for category, pattern in pronoun_patterns.items():
            toxic_count = self.toxic_df[text_column].astype(str).str.contains(
                pattern, case=False, regex=True
            ).sum()
            nontoxic_count = self.nontoxic_df[text_column].astype(str).str.contains(
                pattern, case=False, regex=True
            ).sum()
            
            toxic_pct = (toxic_count / len(self.toxic_df)) * 100
            nontoxic_pct = (nontoxic_count / len(self.nontoxic_df)) * 100
            ratio = toxic_pct / nontoxic_pct if nontoxic_pct > 0 else float('inf')
            
            results[category] = {
                'toxic_pct': toxic_pct,
                'nontoxic_pct': nontoxic_pct,
                'ratio': ratio
            }
            
            if not silent:
                print(f"{category:<25} {toxic_pct:.2f}%      {nontoxic_pct:.2f}%        {ratio:.2f}x")
        
        if not silent:
            print(f"\nINTERPRETATION:")
            if results['second_person']['ratio'] > 2:
                print(f"  ✓ Toxic comments are {results['second_person']['ratio']:.2f}x more likely to use 'you'")
                print(f"    This suggests toxic speech often DIRECTLY TARGETS other players")
        
        return results
    
    def analyze_punctuation_patterns(self):
        """
        Objective 3: Statistical pattern analysis - punctuation usage
        
        Analyzes punctuation patterns that might indicate emotional intensity:
        - Multiple exclamation marks (!!!)
        - Multiple question marks (???)
        - Combined punctuation (!?)
        """
        print("\n"  )
        print("ANALYSIS 5: PUNCTUATION PATTERNS")
        
        punctuation_patterns = {
            'multiple_exclamation': r'!{2,}',
            'multiple_question': r'\?{2,}',
            'mixed_punctuation': r'[!?]{2,}',
            'ellipsis': r'\.{3,}'
        }
        
        # Use original_text for punctuation analysis
        text_column = 'original_text' if 'original_text' in self.df.columns else 'cleaned_text'
        
        print(f"\nPUNCTUATION USAGE (% of messages containing each pattern):")
        print(f"{'Pattern':<25} {'Toxic %':<12} {'Non-toxic %':<15} {'Ratio':<10}")
        print("-" * 65)
        
        results = {}
        for category, pattern in punctuation_patterns.items():
            toxic_count = self.toxic_df[text_column].astype(str).str.contains(
                pattern, case=False, regex=True
            ).sum()
            nontoxic_count = self.nontoxic_df[text_column].astype(str).str.contains(
                pattern, case=False, regex=True
            ).sum()
            
            toxic_pct = (toxic_count / len(self.toxic_df)) * 100
            nontoxic_pct = (nontoxic_count / len(self.nontoxic_df)) * 100
            ratio = toxic_pct / nontoxic_pct if nontoxic_pct > 0 else float('inf')
            
            results[category] = {
                'toxic_pct': toxic_pct,
                'nontoxic_pct': nontoxic_pct,
                'ratio': ratio
            }
            
            print(f"{category:<25} {toxic_pct:.2f}%      {nontoxic_pct:.2f}%        {ratio:.2f}x")
        
        return results
    
    def analyze_ngrams(self, n=2, top_k=20, silent=False):
        """
        Objective 3: Statistical pattern analysis - common phrases
        
        Analyzes common word sequences (n-grams) to identify:
        - Bigrams (2-word phrases): Common insult patterns
        - Trigrams (3-word phrases): More complex toxic patterns
        
        This helps understand the specific phrases used in toxic communication.
        """
        if not silent:
            print(f"\n"  )
            print(f"ANALYSIS 6: COMMON {n}-GRAM PHRASES")
              
        
        def get_top_ngrams(texts, n, top_k):
            """Extract most common n-grams from a list of texts"""
            if len(texts) == 0:
                return []
            
            # Configure vectorizer to extract n-grams
            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                stop_words='english',
                max_features=top_k,
                lowercase=True
            )
            
            try:
                X = vectorizer.fit_transform(texts.dropna().astype(str))
                sums = X.sum(axis=0).A1  # Convert to 1D array
                words = vectorizer.get_feature_names_out()
                
                # Sort by frequency
                sorted_items = sorted(zip(words, sums), key=lambda x: x[1], reverse=True)
                return sorted_items[:top_k]
            except Exception as e:
                if not silent:
                    print(f"  Error extracting n-grams: {e}")
                return []
        
        # Use cleaned_text for n-gram analysis to avoid punctuation splitting
        text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'original_text'
        
        # Get n-grams for toxic and non-toxic comments
        if not silent:
            print(f"\nExtracting top {n}-grams...")
        
        toxic_ngrams = get_top_ngrams(self.toxic_df[text_column], n, top_k)
        nontoxic_ngrams = get_top_ngrams(self.nontoxic_df[text_column], n, top_k)
        
        if not silent:
            print(f"\nTOP {n}-GRAMS IN TOXIC COMMENTS:")
            if toxic_ngrams:
                for i, (ngram, count) in enumerate(toxic_ngrams[:10], 1):
                    print(f"  {i:2d}. '{ngram}': {int(count)} occurrences")
            else:
                print("  No common phrases found")
            
            print(f"\nTOP {n}-GRAMS IN NON-TOXIC COMMENTS:")
            if nontoxic_ngrams:
                for i, (ngram, count) in enumerate(nontoxic_ngrams[:10], 1):
                    print(f"  {i:2d}. '{ngram}': {int(count)} occurrences")
            else:
                print("  No common phrases found")
        
        return toxic_ngrams, nontoxic_ngrams
    
    def generate_wordclouds(self, output_folder):
        """
        Objective 4: Visualize toxicity trends
        
        Creates word clouds to visually represent the most common words in:
        - Toxic comments (red/black theme)
        - Non-toxic comments (green/white theme)
        
        Word clouds provide an intuitive visualization of linguistic patterns.
        """
        print("\n"  )
        print("VISUALIZATION 1: GENERATING WORD CLOUDS")
          
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Use cleaned_text for word clouds to avoid punctuation clutter
        text_column = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'original_text'
        
        # Toxic word cloud
        toxic_text = ' '.join(self.toxic_df[text_column].dropna().astype(str))
        if toxic_text and len(toxic_text.split()) > 10:
            wordcloud_toxic = WordCloud(
                width=1200, height=600,
                background_color='black',
                colormap='Reds',  # Red theme for toxic
                stopwords=stop_words,
                max_words=100,
                random_state=42
            ).generate(toxic_text)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud_toxic, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Toxic Comments', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'wordcloud_toxic.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Toxic word cloud saved")
        
        # Non-toxic word cloud
        nontoxic_text = ' '.join(self.nontoxic_df[text_column].dropna().astype(str))
        if nontoxic_text and len(nontoxic_text.split()) > 10:
            wordcloud_nontoxic = WordCloud(
                width=1200, height=600,
                background_color='white',
                colormap='Greens',  # Green theme for non-toxic
                stopwords=stop_words,
                max_words=100,
                random_state=42
            ).generate(nontoxic_text)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud_nontoxic, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Words in Non-Toxic Comments', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'wordcloud_nontoxic.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Non-toxic word cloud saved")
    
    def create_comparison_dashboard(self, output_folder):
        """
        Objective 4: Visualize toxicity trends and classification outcomes
        
        Creates a comprehensive dashboard with multiple visualizations:
        1. Message length distribution
        2. Profanity comparison
        3. Pronoun usage patterns
        4. Top toxic phrases
        5. Top non-toxic phrases
        6. Failure tag distribution (from Stage 1)
        
        This single figure provides an overview of all linguistic patterns.
        """
        print("\n"  )
        print("VISUALIZATION 2: CREATING COMPARISON DASHBOARD")
          
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Linguistic Pattern Analysis: Toxic vs Non-Toxic Comments', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        # 1. Message length distribution
        ax = axes[0, 0]
        if 'word_count' in self.df.columns:
            toxic_words = self.toxic_df['word_count'].dropna()
            nontoxic_words = self.nontoxic_df['word_count'].dropna()
            
            ax.hist([toxic_words, nontoxic_words], bins=30, alpha=0.7, 
                    label=['Toxic', 'Non-toxic'], color=['crimson', 'forestgreen'])
            ax.set_xlabel('Word Count', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Message Length Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_yscale('log')  # Log scale for better visualization
            
            # Add mean lines
            ax.axvline(toxic_words.mean(), color='darkred', linestyle='--', linewidth=2)
            ax.axvline(nontoxic_words.mean(), color='darkgreen', linestyle='--', linewidth=2)
        
        # 2. Profanity comparison
        ax = axes[0, 1]
        if 'has_profanity' in self.df.columns:
            profanity_data = pd.DataFrame({
                'No Profanity': [
                    (self.toxic_df['has_profanity'] == False).sum() / len(self.toxic_df) * 100,
                    (self.nontoxic_df['has_profanity'] == False).sum() / len(self.nontoxic_df) * 100
                ],
                'Has Profanity': [
                    (self.toxic_df['has_profanity'] == True).sum() / len(self.toxic_df) * 100,
                    (self.nontoxic_df['has_profanity'] == True).sum() / len(self.nontoxic_df) * 100
                ]
            }, index=['Toxic', 'Non-toxic'])
            
            profanity_data.plot(kind='bar', ax=ax, color=['lightgray', 'crimson'])
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title('Profanity Presence', fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', fontsize=10)
        
        # 3. Pronoun usage
        ax = axes[0, 2]
        pronoun_results = self.analyze_pronoun_usage(silent=True)
        
        if pronoun_results:
            pronoun_df = pd.DataFrame({
                'Toxic': [v['toxic_pct'] for v in pronoun_results.values()],
                'Non-toxic': [v['nontoxic_pct'] for v in pronoun_results.values()]
            }, index=[k.replace('_', ' ').title() for k in pronoun_results.keys()])
            
            pronoun_df.plot(kind='bar', ax=ax, color=['crimson', 'forestgreen'])
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title('Pronoun Usage Patterns', fontsize=14, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend()
        
        # 4. Top toxic bigrams
        ax = axes[1, 0]
        toxic_bigrams, _ = self.analyze_ngrams(n=2, top_k=20, silent=True)
        
        if toxic_bigrams:
            words, counts = zip(*toxic_bigrams[:10])
            y_pos = np.arange(len(words))
            
            bars = ax.barh(y_pos, counts, color='crimson')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_title('Top 10 Toxic Phrases (Bigrams)', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count + 5, bar.get_y() + bar.get_height()/2, 
                       str(int(count)), va='center', fontsize=9)
        
        # 5. Top non-toxic bigrams
        ax = axes[1, 1]
        _, nontoxic_bigrams = self.analyze_ngrams(n=2, top_k=20, silent=True)
        
        if nontoxic_bigrams:
            words, counts = zip(*nontoxic_bigrams[:10])
            y_pos = np.arange(len(words))
            
            bars = ax.barh(y_pos, counts, color='forestgreen')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_title('Top 10 Non-Toxic Phrases (Bigrams)', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count + 5, bar.get_y() + bar.get_height()/2, 
                       str(int(count)), va='center', fontsize=9)
        
        # 6. Failure tags distribution (from Stage 1)
        ax = axes[1, 2]
        if 'failure_tags' in self.df.columns:
            # Parse failure tags
            all_tags = []
            toxic_tags_list = []
            nontoxic_tags_list = []
            
            for idx, row in self.df.iterrows():
                tags = row['failure_tags']
                if isinstance(tags, str):
                    try:
                        tags = ast.literal_eval(tags)
                    except:
                        tags = []
                elif not isinstance(tags, list):
                    tags = []
                
                all_tags.extend(tags)
                if row['toxic_label'] == 1:
                    toxic_tags_list.extend(tags)
                else:
                    nontoxic_tags_list.extend(tags)
            
            if all_tags:
                toxic_tag_counts = pd.Series(toxic_tags_list).value_counts()
                nontoxic_tag_counts = pd.Series(nontoxic_tags_list).value_counts()
                
                # Create comparison dataframe
                all_categories = list(set(toxic_tag_counts.index) | set(nontoxic_tag_counts.index))
                tag_comparison = pd.DataFrame(index=all_categories)
                tag_comparison['Toxic'] = toxic_tag_counts
                tag_comparison['Non-toxic'] = nontoxic_tag_counts
                tag_comparison = tag_comparison.fillna(0)
                
                # Convert to percentages
                for col in tag_comparison.columns:
                    tag_comparison[col] = tag_comparison[col] / tag_comparison[col].sum() * 100
                
                tag_comparison = tag_comparison.sort_values('Toxic', ascending=False).head(8)
                
                tag_comparison.plot(kind='bar', ax=ax, color=['crimson', 'forestgreen'])
                ax.set_title('Failure Tags Distribution', fontsize=14, fontweight='bold')
                ax.set_ylabel('Percentage (%)', fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'linguistic_dashboard.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Comparison dashboard saved")

def main():
    """
    Main execution function for Stage 3 linguistic analysis.
    
    This pipeline:
    1. Loads data from Stage 1/2
    2. Performs comprehensive linguistic pattern analysis
    3. Generates visualizations for the thesis
    4. Saves all results to output folder
    """
    print("\n"  )
    print("STAGE 3: LINGUISTIC PATTERN ANALYSIS")
      
    print("\nThis stage addresses Thesis Objectives 3 & 4:")
    print("  O3: Conduct data analysis to identify linguistic and statistical patterns")
    print("  O4: Visualize toxicity trends and classification outcomes\n")
    
    # First try Stage 2 predictions (has original text)
    input_file = "../output/stage2_outputs/stage2_predictions.csv.gz"
    
    if not os.path.exists(input_file):
        # Fall back to Stage 1 if Stage 2 doesn't exist
        input_file = "../output/stage1_outputs/stage1_output.csv"
        print(f"Stage 2 not found, using Stage 1 data: {input_file}")
    else:
        print(f"Loading Stage 2 predictions: {input_file}")
    
    try:
        if input_file.endswith('.gz'):
            df = pd.read_csv(input_file, compression='gzip')
        else:
            df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file. Please run Stage 1 first.")
        sys.exit(1)
    
    print(f"\nLoaded {len(df):,} comments")
    
    # Handle column naming differences between Stage 1 and Stage 2
    if 'toxic_label' not in df.columns:
        if 'true_label' in df.columns:
            print("Renaming 'true_label' to 'toxic_label'")
            df['toxic_label'] = df['true_label']
        else:
            print("Error: No toxicity label column found!")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Handle text column - prioritize original text for analysis
    if 'text' in df.columns:
        print("Using original 'text' column for analysis (preserves punctuation, caps, etc.)")
        df['original_text'] = df['text']
        # Also create a cleaned version for word clouds
        df['cleaned_text'] = df['text'].astype(str).str.lower()
    elif 'comment_text' in df.columns:
        print("Using 'comment_text' column")
        df['original_text'] = df['comment_text']
        df['cleaned_text'] = df['comment_text'].astype(str).str.lower()
    elif 'cleaned_text' in df.columns:
        print("Using 'cleaned_text' column (note: punctuation may be removed)")
        df['original_text'] = df['cleaned_text']  # Use same for both
    else:
        print("Error: No text column found!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Toxic comments: {df['toxic_label'].sum():,} ({df['toxic_label'].mean()*100:.2f}%)")
    
    # Parse failure tags if they exist
    if 'failure_tags' in df.columns:
        print("Parsing failure tags...")
        df['failure_tags'] = df['failure_tags'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else 
                     (x if isinstance(x, list) else [])
        )
    
    # Initialize analyzer
    analyzer = LinguisticAnalyzer(df)
    
    # Create output folder
    output_folder = "../output/stage3_outputs"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nOutput will be saved to: {output_folder}")
    
    # Run all analyses
    results = {}
    
    # 1. Message length analysis
    results['length'] = analyzer.analyze_message_length()
    
    # 2. Vocabulary diversity
    results['vocabulary'] = analyzer.analyze_vocabulary_diversity()
    
    # 3. Profanity patterns
    results['profanity'] = analyzer.analyze_profanity_patterns()
    
    # 4. Pronoun usage
    results['pronouns'] = analyzer.analyze_pronoun_usage()
    
    # 5. Punctuation patterns
    results['punctuation'] = analyzer.analyze_punctuation_patterns()
    
    # 6. N-gram analysis (bigrams and trigrams)
    results['bigrams'] = analyzer.analyze_ngrams(n=2, top_k=20)
    results['trigrams'] = analyzer.analyze_ngrams(n=3, top_k=20)
    
    # Generate visualizations
    analyzer.generate_wordclouds(output_folder)
    analyzer.create_comparison_dashboard(output_folder)
    
    # Save all results to CSV
    print("\n"  )
    print("SAVING RESULTS")
      
    
    # Flatten results for CSV
    flat_results = []
    
    # Length results
    flat_results.append({'metric': 'toxic_avg_word_count', 'value': results['length']['toxic_word_avg']})
    flat_results.append({'metric': 'nontoxic_avg_word_count', 'value': results['length']['nontoxic_word_avg']})
    flat_results.append({'metric': 'toxic_avg_char_count', 'value': results['length']['toxic_char_avg']})
    flat_results.append({'metric': 'nontoxic_avg_char_count', 'value': results['length']['nontoxic_char_avg']})
    flat_results.append({'metric': 'length_t_statistic', 'value': results['length']['t_stat']})
    flat_results.append({'metric': 'length_p_value', 'value': results['length']['p_value']})
    
    # Vocabulary results
    flat_results.append({'metric': 'toxic_ttr', 'value': results['vocabulary']['toxic_ttr']})
    flat_results.append({'metric': 'nontoxic_ttr', 'value': results['vocabulary']['nontoxic_ttr']})
    
    # Profanity results (overall)
    if 'has_profanity' in df.columns:
        toxic_prof_pct = (analyzer.toxic_df['has_profanity'].sum() / len(analyzer.toxic_df)) * 100
        nontoxic_prof_pct = (analyzer.nontoxic_df['has_profanity'].sum() / len(analyzer.nontoxic_df)) * 100
        flat_results.append({'metric': 'toxic_profanity_pct', 'value': toxic_prof_pct})
        flat_results.append({'metric': 'nontoxic_profanity_pct', 'value': nontoxic_prof_pct})
    
    # Save to CSV
    results_df = pd.DataFrame(flat_results)
    results_df.to_csv(os.path.join(output_folder, 'linguistic_metrics.csv'), index=False)
    print("  ✓ Linguistic metrics saved to CSV")
    
    # Save detailed profanity breakdown
    if 'profanity' in results and results['profanity']:
        profanity_df = pd.DataFrame(results['profanity']).T
        profanity_df.to_csv(os.path.join(output_folder, 'profanity_breakdown.csv'))
        print("  ✓ Profanity breakdown saved")
    
    # Generate summary report
    with open(os.path.join(output_folder, 'summary_report.txt'), 'w') as f:
        f.write("STAGE 3: LINGUISTIC PATTERN ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("THESIS OBJECTIVES ADDRESSED:\n")
        f.write("-"*40 + "\n")
        f.write("O3: Conduct data analysis to identify linguistic and statistical patterns\n")
        f.write("O4: Visualize toxicity trends and classification outcomes\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*40 + "\n\n")
        
        f.write("1. MESSAGE LENGTH PATTERNS:\n")
        f.write(f"   - Toxic comments: {results['length']['toxic_word_avg']:.2f} words on average\n")
        f.write(f"   - Non-toxic comments: {results['length']['nontoxic_word_avg']:.2f} words on average\n")
        f.write(f"   - {'Toxic' if results['length']['toxic_word_avg'] > results['length']['nontoxic_word_avg'] else 'Non-toxic'} comments tend to be longer\n")
        f.write(f"   - Statistical significance: p-value = {results['length']['p_value']:.4f}\n\n")
        
        f.write("2. VOCABULARY DIVERSITY:\n")
        f.write(f"   - Toxic TTR: {results['vocabulary']['toxic_ttr']:.4f}\n")
        f.write(f"   - Non-toxic TTR: {results['vocabulary']['nontoxic_ttr']:.4f}\n")
        f.write(f"   - {'Toxic' if results['vocabulary']['toxic_ttr'] > results['vocabulary']['nontoxic_ttr'] else 'Non-toxic'} comments use more diverse vocabulary\n\n")
        
        if 'has_profanity' in df.columns:
            f.write("3. PROFANITY USAGE:\n")
            f.write(f"   - Toxic comments with profanity: {toxic_prof_pct:.2f}%\n")
            f.write(f"   - Non-toxic comments with profanity: {nontoxic_prof_pct:.2f}%\n")
            f.write(f"   - Toxic comments are {toxic_prof_pct/nontoxic_prof_pct:.2f}x more likely to contain profanity\n\n")
        
        f.write("4. PRONOUN PATTERNS:\n")
        for pronoun, values in results['pronouns'].items():
            f.write(f"   - {pronoun.replace('_', ' ').title()}:\n")
            f.write(f"       Toxic: {values['toxic_pct']:.2f}%\n")
            f.write(f"       Non-toxic: {values['nontoxic_pct']:.2f}%\n")
            f.write(f"       Ratio: {values['ratio']:.2f}x\n")
        f.write("\n")
        
        f.write("5. PUNCTUATION PATTERNS:\n")
        for punct, values in results['punctuation'].items():
            f.write(f"   - {punct.replace('_', ' ').title()}:\n")
            f.write(f"       Toxic: {values['toxic_pct']:.2f}%\n")
            f.write(f"       Non-toxic: {values['nontoxic_pct']:.2f}%\n")
        f.write("\n")
        
        f.write("6. COMMON PHRASES:\n")
        if results['bigrams'][0]:
            f.write("   Top toxic bigrams:\n")
            for i, (phrase, count) in enumerate(results['bigrams'][0][:5], 1):
                f.write(f"     {i}. '{phrase}' ({int(count)} occurrences)\n")
        f.write("\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-"*40 + "\n")
        f.write("linguistic_metrics.csv - Quantitative measurements\n")
        f.write("profanity_breakdown.csv - Detailed profanity statistics\n")
        f.write("wordcloud_toxic.png - Word cloud of toxic comments\n")
        f.write("wordcloud_nontoxic.png - Word cloud of non-toxic comments\n")
        f.write("linguistic_dashboard.png - Comprehensive visualization dashboard\n")
        f.write("summary_report.txt - This summary file\n")
    
    print("\n"  )
    print("STAGE 3 COMPLETED SUCCESSFULLY!")
      
    print(f"\nAll outputs saved to: {output_folder}")
    print("\nFiles generated:")
    print("  • linguistic_metrics.csv - Quantitative measurements")
    print("  • profanity_breakdown.csv - Detailed profanity statistics")
    print("  • wordcloud_toxic.png - Toxic word cloud")
    print("  • wordcloud_nontoxic.png - Non-toxic word cloud")
    print("  • linguistic_dashboard.png - Comprehensive dashboard")
    print("  • summary_report.txt - Text summary")

if __name__ == "__main__":
    main()