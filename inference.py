# inference.py - Experiment with trained model
import torch
import torch.nn as nn
import pickle
import numpy as np
from model import create_model
import time
import random

class ModelInference:
    def __init__(self, model_path="best_model.pth", force_cpu=False):
        """
        Initialize the inference class
        
        Args:
            model_path: Path to the saved model checkpoint
            force_cpu: If True, forces model to run on CPU even if CUDA is available
        """
        self.force_cpu = force_cpu
        
        # Device selection
        if force_cpu:
            self.device = torch.device('cpu')
            print("🔧 Forcing CPU usage")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Using device: {self.device}")
        
        # Load model and vocab
        self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the trained model and vocabulary"""
        try:
            print("📂 Loading model checkpoint...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load vocabulary
            self.vocab = checkpoint['vocab']
            self.config = checkpoint.get('config', {})
            
            print(f"✅ Model trained for {checkpoint['epoch'] + 1} epochs")
            print(f"📊 Best validation loss: {checkpoint['valid_loss']:.3f}")
            
            # Create model with same config
            self.model = create_model(
                vocab=self.vocab,
                device=self.device,
                embedding_dim=self.config.get('embedding_dim', 256),
                hidden_size=self.config.get('hidden_size', 512),
                enc_layers=self.config.get('enc_layers', 2),
                dec_layers=self.config.get('dec_layers', 4),
                dropout=self.config.get('dropout', 0.3)
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Extract mappings
            self.src_char2idx = self.vocab['src_char2idx']
            self.src_idx2char = self.vocab['src_idx2char']
            self.tgt_char2idx = self.vocab['tgt_char2idx']
            self.tgt_idx2char = self.vocab['tgt_idx2char']
            self.special_tokens = self.vocab['special_tokens']
            
            print(f"📚 Vocabulary sizes: Source={len(self.src_char2idx)}, Target={len(self.tgt_char2idx)}")
            print("✅ Model loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def encode_sequence(self, text, vocab_type='src'):
        """Convert text to tensor of indices"""
        if vocab_type == 'src':
            char2idx = self.src_char2idx
        else:
            char2idx = self.tgt_char2idx
            
        # Convert characters to indices
        indices = []
        for char in text.lower():
            if char in char2idx:
                indices.append(char2idx[char])
            else:
                indices.append(char2idx.get('<UNK>', 1))  # Unknown token
        
        # Add SOS and EOS tokens for target sequences
        if vocab_type == 'tgt':
            indices = [self.special_tokens['SOS']] + indices + [self.special_tokens['EOS']]
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def decode_sequence(self, tensor, vocab_type='tgt'):
        """Convert tensor of indices back to text"""
        if vocab_type == 'src':
            idx2char = self.src_idx2char
        else:
            idx2char = self.tgt_idx2char
        
        # Remove batch dimension if present
        if tensor.dim() > 1:
            tensor = tensor.squeeze(0)
        
        chars = []
        for idx in tensor:
            idx_val = idx.item() if torch.is_tensor(idx) else idx
            if idx_val == self.special_tokens.get('EOS', 3):
                break
            elif idx_val in [self.special_tokens.get('SOS', 2), self.special_tokens.get('PAD', 0)]:
                continue
            else:
                chars.append(idx2char.get(idx_val, '<UNK>'))
        
        return ''.join(chars)
    
    def translate(self, text, max_length=50, beam_search=False, beam_width=3):
        """
        Translate input text using the trained model
        
        Args:
            text: Input text to translate
            max_length: Maximum output length
            beam_search: Whether to use beam search (more accurate but slower)
            beam_width: Width of beam search
        
        Returns:
            translated_text, confidence_score
        """
        if beam_search:
            return self._translate_beam_search(text, max_length, beam_width)
        else:
            return self._translate_greedy(text, max_length)
    
    def _translate_greedy(self, text, max_length):
        """Greedy translation (faster)"""
        with torch.no_grad():
            # Encode input
            src_tensor = self.encode_sequence(text, 'src')
            src_length = torch.tensor([src_tensor.shape[1]], dtype=torch.long).to(self.device)
            
            # Get encoder output
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_length)
            
            # Prepare decoder hidden states
            hidden, cell = self.model._prepare_decoder_hidden(hidden, cell)
            
            # Start with SOS token
            decoder_input = torch.tensor([[self.special_tokens['SOS']]], dtype=torch.long).to(self.device)
            decoded_chars = []
            confidences = []
            
            for _ in range(max_length):
                output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)
                
                # Get probabilities and prediction
                probs = torch.softmax(output.squeeze(1), dim=-1)
                predicted_idx = output.argmax(2)
                confidence = probs.max().item()
                
                predicted_char_idx = predicted_idx.item()
                
                # Stop if EOS token
                if predicted_char_idx == self.special_tokens.get('EOS', 3):
                    break
                
                # Add character to result
                char = self.tgt_idx2char.get(predicted_char_idx, '<UNK>')
                decoded_chars.append(char)
                confidences.append(confidence)
                
                # Next input
                decoder_input = predicted_idx
            
            translated_text = ''.join(decoded_chars)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return translated_text, avg_confidence
    
    def _translate_beam_search(self, text, max_length, beam_width):
        """Beam search translation (more accurate but slower)"""
        with torch.no_grad():
            # Encode input
            src_tensor = self.encode_sequence(text, 'src')
            src_length = torch.tensor([src_tensor.shape[1]], dtype=torch.long).to(self.device)
            
            # Get encoder output
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor, src_length)
            hidden, cell = self.model._prepare_decoder_hidden(hidden, cell)
            
            # Initialize beams
            beams = [{'sequence': [self.special_tokens['SOS']], 'score': 0.0, 'hidden': hidden, 'cell': cell}]
            completed_beams = []
            
            for step in range(max_length):
                new_beams = []
                
                for beam in beams:
                    if beam['sequence'][-1] == self.special_tokens.get('EOS', 3):
                        completed_beams.append(beam)
                        continue
                    
                    decoder_input = torch.tensor([[beam['sequence'][-1]]], dtype=torch.long).to(self.device)
                    output, new_hidden, new_cell = self.model.decoder(decoder_input, beam['hidden'], beam['cell'])
                    
                    # Get top k predictions
                    log_probs = torch.log_softmax(output.squeeze(1), dim=-1)
                    top_scores, top_indices = log_probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        new_sequence = beam['sequence'] + [top_indices[0][i].item()]
                        new_score = beam['score'] + top_scores[0][i].item()
                        
                        new_beams.append({
                            'sequence': new_sequence,
                            'score': new_score,
                            'hidden': new_hidden,
                            'cell': new_cell
                        })
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x['score'] / len(x['sequence']), reverse=True)[:beam_width]
                
                if len(completed_beams) >= beam_width:
                    break
            
            # Get best completed beam or best current beam
            all_beams = completed_beams + beams
            best_beam = max(all_beams, key=lambda x: x['score'] / len(x['sequence']))
            
            # Decode sequence (skip SOS)
            decoded_chars = []
            for idx in best_beam['sequence'][1:]:  # Skip SOS
                if idx == self.special_tokens.get('EOS', 3):
                    break
                char = self.tgt_idx2char.get(idx, '<UNK>')
                decoded_chars.append(char)
            
            translated_text = ''.join(decoded_chars)
            confidence = np.exp(best_beam['score'] / len(best_beam['sequence']))
            
            return translated_text, confidence
    
    def benchmark_speed(self, test_texts, num_runs=3):
        """Benchmark inference speed"""
        print(f"🏃‍♂️ Benchmarking inference speed on {self.device}...")
        
        times_greedy = []
        times_beam = []
        
        for run in range(num_runs):
            # Greedy decoding
            start_time = time.time()
            for text in test_texts:
                self.translate(text, beam_search=False)
            greedy_time = time.time() - start_time
            times_greedy.append(greedy_time)
            
            # Beam search
            start_time = time.time()
            for text in test_texts:
                self.translate(text, beam_search=True, beam_width=3)
            beam_time = time.time() - start_time
            times_beam.append(beam_time)
        
        avg_greedy = np.mean(times_greedy)
        avg_beam = np.mean(times_beam)
        
        print(f"📊 Average time for {len(test_texts)} translations:")
        print(f"   Greedy decoding: {avg_greedy:.3f}s ({avg_greedy/len(test_texts):.3f}s per text)")
        print(f"   Beam search: {avg_beam:.3f}s ({avg_beam/len(test_texts):.3f}s per text)")
        print(f"   Speedup (Greedy): {avg_beam/avg_greedy:.1f}x faster\n")

def main():
    """Main function to demonstrate the model using real test samples"""
    
    print("="*60)
    print("🎯 URDU-ROMAN TRANSLATION MODEL INFERENCE")
    print("="*60)
    
    try:
        # Initialize inference (try GPU first, fallback to CPU)
        inference = ModelInference("best_model.pth", force_cpu=False)
        
        # Load tensor_pairs.pkl to get real test samples
        print("📂 Loading test samples from tensor_pairs.pkl...")
        try:
            with open("tensor_pairs.pkl", "rb") as f:
                tensor_pairs = pickle.load(f)
            
            # Assume tensor_pairs is a list of (src_tensor, tgt_tensor)
            # We'll take the first 10 source sequences as test inputs
            test_src_tensors = [pair[0] for pair in tensor_pairs[:10]]
            
            # Decode source tensors to text using the loaded vocab
            test_cases = []
            for src_tensor in test_src_tensors:
                # src_tensor is likely a 1D tensor of indices
                if src_tensor.dim() == 2:
                    src_tensor = src_tensor.squeeze(0)
                text = inference.decode_sequence(src_tensor, vocab_type='src')
                if text.strip():  # avoid empty strings
                    test_cases.append(text)
            
            if not test_cases:
                raise ValueError("No valid test cases decoded from tensor_pairs.pkl")
                
            print(f"✅ Loaded {len(test_cases)} test samples from dataset.")
            
        except Exception as e:
            print(f"⚠️  Could not load tensor_pairs.pkl: {e}")
            print("🔄 Falling back to default test cases...")
            test_cases = [
                "salam", "kya haal hai", "main thik hun", "tumhara naam", "khana khao",
                "paani piyo", "kitab parho", "school jao", "ghar aao", "so jao"
            ]
        
        print("\n🧪 TESTING TRANSLATIONS:")
        print("-"*40)
        
        print("🎯 GREEDY DECODING:")
        for i, text in enumerate(test_cases, 1):
            translation, confidence = inference.translate(text, beam_search=False)
            print(f"{i:2d}. Input:  '{text}'")
            print(f"    Output: '{translation}' (confidence: {confidence:.3f})")
            print()
        
        print("\n" + "="*60)
        print("🎯 BEAM SEARCH DECODING:")
        print("-"*40)
        
        # Test a few with beam search for comparison
        for i, text in enumerate(test_cases[:5], 1):
            translation, confidence = inference.translate(text, beam_search=True, beam_width=3)
            print(f"{i:2d}. Input:  '{text}'")
            print(f"    Output: '{translation}' (confidence: {confidence:.3f})")
            print()
        
        # Speed benchmark
        print("="*60)
        inference.benchmark_speed(test_cases[:5])
        
        # Interactive mode
        print("🎮 INTERACTIVE MODE:")
        print("Enter text to translate (or 'quit' to exit):")
        print("-"*40)
        
        while True:
            try:
                user_input = input("Input: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    greedy_result, greedy_conf = inference.translate(user_input, beam_search=False)
                    beam_result, beam_conf = inference.translate(user_input, beam_search=True)
                    
                    print(f"Greedy:    '{greedy_result}' (conf: {greedy_conf:.3f})")
                    print(f"Beam:      '{beam_result}' (conf: {beam_conf:.3f})")
                    print()
                    
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thanks for using the translator!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure 'best_model.pth' exists in the current directory")
        print("2. Ensure 'tensor_pairs.pkl' is available for test samples")
        print("3. Check that vocab.pkl is also available (should be in model checkpoint)")
        
        # Try CPU-only mode
        print("\n🔄 Trying CPU-only mode...")
        try:
            inference = ModelInference("best_model.pth", force_cpu=True)
            result, conf = inference.translate("salam", beam_search=False)
            print(f"✅ CPU mode works! Test translation: 'salam' -> '{result}'")
        except Exception as cpu_error:
            print(f"❌ CPU mode also failed: {cpu_error}")

def compare_cpu_gpu_speed():
    """Compare inference speed between CPU and GPU"""
    test_text = "salam kya haal hai"
    
    print("⚡ COMPARING CPU vs GPU PERFORMANCE:")
    print("="*50)
    
    # Test GPU
    if torch.cuda.is_available():
        try:
            print("🚀 Testing GPU performance...")
            gpu_inference = ModelInference("best_model.pth", force_cpu=False)
            
            start_time = time.time()
            for _ in range(10):
                gpu_inference.translate(test_text)
            gpu_time = time.time() - start_time
            
            print(f"   GPU: {gpu_time:.3f}s for 10 translations")
        except Exception as e:
            print(f"   GPU failed: {e}")
            gpu_time = float('inf')
    else:
        print("   GPU: Not available")
        gpu_time = float('inf')
    
    # Test CPU
    try:
        print("🖥️  Testing CPU performance...")
        cpu_inference = ModelInference("best_model.pth", force_cpu=True)
        
        start_time = time.time()
        for _ in range(10):
            cpu_inference.translate(test_text)
        cpu_time = time.time() - start_time
        
        print(f"   CPU: {cpu_time:.3f}s for 10 translations")
        
        if gpu_time != float('inf'):
            speedup = cpu_time / gpu_time
            print(f"🏁 GPU is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than CPU")
        
    except Exception as e:
        print(f"   CPU failed: {e}")


if __name__ == "__main__":
    main()
    
    # Uncomment to run performance comparison
    # print("\n" + "="*60)
    # compare_cpu_gpu_speed()