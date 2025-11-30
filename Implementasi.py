import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
print("Memuat model tomato_disease_efficientnet.keras...")
model = tf.keras.models.load_model('Tomato97.keras')
print("Model berhasil dimuat!")

# Daftar kelas penyakit daun tomat
class_names = [
    "Blight",
    "Healthy", 
    "Leaf_Mold",
    "Spider_Mites",
    "Yellow_Leaf_Curl_Virus"
]

# Preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array

# Memprediksi Penyakit dari Input
def predict_tomato_disease(image_path):
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get all probabilities
        class_probabilities = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        return predicted_class, confidence, class_probabilities
        
    except Exception as e:
        print(f"Error dalam memproses gambar: {e}")
        return None, None, None

# Menampilkan hasil prediksi
def display_prediction(image_path, predicted_class, confidence, class_probabilities):
    original_img = image.load_img(image_path)
    
    plt.figure(figsize=(12, 6))
    
    # Plot gambar
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f'Gambar Input\nPredicted: {predicted_class}\nConfidence: {confidence:.4f}')
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    classes = list(class_probabilities.keys())
    probabilities = list(class_probabilities.values())
    
    colors = ['red' if prob == max(probabilities) else 'blue' for prob in probabilities]
    bars = plt.barh(classes, probabilities, color=colors)
    plt.xlabel('Probability')
    plt.title('Probabilitas Kelas')
    plt.xlim(0, 1)
    
    # Menampilkan nilai Probability
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, 
                f'{prob:.4f}', ha='right', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Main Function
def main():
    print("=" * 60)
    print("TOMATO DISEASE CLASSIFIER")
    print("=" * 60)
    print(f"Kelas yang dapat diprediksi: {', '.join(class_names)}")
    print()
    
    while True:
        print("\nPilih opsi:")
        print("1. Prediksi gambar dari path")
        print("2. Keluar")
        
        choice = input("Masukkan pilihan (1/2): ").strip()
        
        if choice == '1':
            image_path = input("Masukkan path lengkap gambar: ").strip()
            image_path = image_path.strip('"').strip("'")
            
            try:
                # Proses Prediksi
                predicted_class, confidence, probabilities = predict_tomato_disease(image_path)
                
                if predicted_class is not None:
                    # Menampilkan hasil
                    print("\n" + "="*40)
                    print("HASIL PREDIKSI:")
                    print("="*40)
                    print(f"Gambar: {image_path}")
                    print(f"Predicted Class: {predicted_class}")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"Confidence (%): {confidence*100:.2f}%")
                    
                    print("\nProbabilitas semua kelas:")
                    for class_name, prob in probabilities.items():
                        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
                    
                    # Menampilkan visualisasi plot
                    display_prediction(image_path, predicted_class, confidence, probabilities)
                        
                else:
                    print("Gagal melakukan prediksi. Pastikan path gambar benar.")
                    
            except Exception as e:
                print(f"Error: {e}")
                print("Pastikan path gambar benar dan format gambar didukung (jpg, png, jpeg)")
                
        elif choice == '2':
            print("Terima kasih telah menggunakan Tomato Disease Classifier!")
            break
            
        else:
            print("Pilihan tidak valid. Silakan pilih 1 atau 2.")

if __name__ == "__main__":
    main()