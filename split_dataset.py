import os
import shutil
from sklearn.model_selection import train_test_split

def dividir_train_test_val(directorio_base, ratio_entrenamiento=0.8, ratio_validacion=0.1, seed=42):
    # Obtener la lista de clases (nombres de las subcarpetas)
    clases = os.listdir(directorio_base)
    
    for clase in clases:
        # Crear las rutas completas para la clase
        ruta_clase = os.path.join(directorio_base, clase)
        
        # Obtener la lista de archivos en la clase
        archivos = os.listdir(ruta_clase)
        
        # Dividir la lista de archivos en entrenamiento y prueba + validación
        train_files, test_val_files = train_test_split(archivos, train_size=ratio_entrenamiento, random_state=seed)
        
        # Dividir la lista de archivos de prueba + validación en prueba y validación
        test_files, val_files = train_test_split(test_val_files, train_size=ratio_validacion / (1 - ratio_entrenamiento), random_state=seed)
        
        # Crear las carpetas de entrenamiento, prueba y validación
        carpeta_entrenamiento = os.path.join(directorio_base, 'train', clase)
        carpeta_prueba = os.path.join(directorio_base, 'test', clase)
        carpeta_validacion = os.path.join(directorio_base, 'val', clase)
        
        os.makedirs(carpeta_entrenamiento, exist_ok=True)
        os.makedirs(carpeta_prueba, exist_ok=True)
        os.makedirs(carpeta_validacion, exist_ok=True)
        
        # Mover archivos a las carpetas correspondientes
        for archivo in train_files:
            shutil.move(os.path.join(ruta_clase, archivo), os.path.join(carpeta_entrenamiento, archivo))
        
        for archivo in test_files:
            shutil.move(os.path.join(ruta_clase, archivo), os.path.join(carpeta_prueba, archivo))
            
        for archivo in val_files:
            shutil.move(os.path.join(ruta_clase, archivo), os.path.join(carpeta_validacion, archivo))

if __name__ == "__main__":
    # Ruta al directorio base que contiene las subcarpetas de clases
    directorio_base = 'data/full_dataset'
    
    # Proporción de imágenes para entrenamiento (por defecto, 80%)
    ratio_entrenamiento = 0.8
    
    # Proporción de imágenes para validación (por defecto, 10%)
    ratio_validacion = 0.1
    
    # Semilla para reproducibilidad
    seed = 42
    
    # Llamar a la función para dividir en entrenamiento, prueba y validación
    dividir_train_test_val(directorio_base, ratio_entrenamiento, ratio_validacion, seed)


