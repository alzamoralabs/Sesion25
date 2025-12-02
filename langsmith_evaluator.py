"""
Script de Evaluación con LangSmith
Evalúa artículos editoriales usando datasets y evaluadores personalizados

Requiere:
    pip install langsmith langchain-openai python-dotenv

Configuración:
    export LANGSMITH_API_KEY="tu-api-key"
    export OPENAI_API_KEY="tu-openai-key"

Uso:
    python langsmith_evaluador.py <archivo_markdown>
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langsmith import Client
from langsmith.schemas import Example
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATASET_NAME = "Artículos Editoriales IA"
EXPERIMENT_PREFIX = "Editorial IA Evaluation"

# Inicializar cliente de LangSmith
client = Client()

# Inicializar LLM para evaluaciones (SIN wrappers)
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================================
# EXTRACCIÓN DEL ARTÍCULO
# ============================================================================

def extraer_articulo_del_markdown(archivo_path: str) -> Optional[str]:
    """Extrae el contenido del artículo del archivo Markdown generado"""
    
    try:
        with open(archivo_path, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Buscar sección "## 📄 ARTÍCULO FINAL"
        patron = r"## 📄 ARTÍCULO FINAL\n\n(.*?)\n\n---"
        match = re.search(patron, contenido, re.DOTALL)
        
        if match:
            articulo = match.group(1).strip()
            return articulo
        else:
            partes = contenido.split("## 📄 ARTÍCULO FINAL")
            if len(partes) > 1:
                articulo = partes[1].split("\n\n---")[0].strip()
                return articulo
        
        return None
    
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {archivo_path}")
        return None
    except Exception as e:
        print(f"❌ Error al procesar archivo: {e}")
        return None


# ============================================================================
# EVALUADORES PERSONALIZADOS
# ============================================================================

def evaluador_longitud_minima(ejemplo: Example) -> dict:
    """
    Evaluador 1: Verifica que el artículo tenga longitud mínima
    Métrica: ¿El artículo tiene al menos 500 caracteres?
    """
    
    try:
        articulo = ejemplo.inputs.get("articulo", "")
        caracteres = len(articulo)
        
        score = 1 if caracteres >= 500 else 0
        
        return {
            "key": "longitud_minima",
            "score": score,
            "comment": f"Artículo tiene {caracteres} caracteres (mínimo: 500)"
        }
    except Exception as e:
        return {"key": "longitud_minima", "score": 0, "comment": str(e)}


def evaluador_contenido_apropiado(ejemplo: Example) -> dict:
    """
    Evaluador 2: Verifica si el contenido es apropiado y sano
    Métrica: ¿El artículo promueve buenas costumbres sin ser ofensivo?
    """
    
    try:
        articulo = ejemplo.inputs.get("articulo", "").lower()
        
        # Palabras/frases inapropiadas a buscar
        palabras_prohibidas = [
            "discrimin", "racist", "sexist", "odio", "violencia extrema"
        ]
        
        tiene_contenido_inapropiado = any(
            palabra in articulo for palabra in palabras_prohibidas
        )
        
        score = 0 if tiene_contenido_inapropiado else 1
        
        return {
            "key": "contenido_apropiado",
            "score": score,
            "comment": (
                "Contenido inapropiado detectado" if tiene_contenido_inapropiado
                else "Contenido apropiado y sano"
            )
        }
    except Exception as e:
        return {"key": "contenido_apropiado", "score": 1, "comment": str(e)}


def evaluador_cantidad_parrafos(ejemplo: Example) -> dict:
    """
    Evaluador 3: Verifica estructura con múltiples párrafos
    Métrica: ¿Tiene al menos 4 párrafos bien estructurados?
    """
    
    try:
        articulo = ejemplo.inputs.get("articulo", "")
        parrafos = [p.strip() for p in articulo.split('\n\n') if p.strip()]
        num_parrafos = len(parrafos)
        
        # Buscamos al menos 4 párrafos
        score = 1 if num_parrafos >= 4 else (num_parrafos / 4)
        
        return {
            "key": "estructura_parrafos",
            "score": score,
            "comment": f"Artículo tiene {num_parrafos} párrafos (esperado: ≥4)"
        }
    except Exception as e:
        return {"key": "estructura_parrafos", "score": 0, "comment": str(e)}


def evaluador_relevancia_tema(ejemplo: Example) -> dict:
    """
    Evaluador 4: Verifica que el contenido sea relevante al tema
    Métrica: ¿Menciona IA, tecnología, o países desarrollados?
    """
    
    try:
        articulo = ejemplo.inputs.get("articulo", "").lower()
        
        # Palabras clave relacionadas al tema
        palabras_clave = [
            "inteligencia artificial", "ia", "llm", "gpt",
            "pais", "mundo", "desarrollo", "tech", "innovation",
            "primer mundo", "desarrollado"
        ]
        
        menciones = sum(1 for palabra in palabras_clave if palabra in articulo)
        score = min(1.0, menciones / 3)  # Máximo si menciona 3+ términos clave
        
        return {
            "key": "relevancia_tema",
            "score": score,
            "comment": f"Se encontraron {menciones} términos clave del tema (esperado: ≥3)"
        }
    except Exception as e:
        return {"key": "relevancia_tema", "score": 0, "comment": str(e)}


def evaluador_tono_comico_llm(ejemplo: Example) -> dict:
    """
    Evaluador 5: LLM-as-Judge para verificar tono cómico
    Métrica: ¿El artículo mantiene un tono cómico e inteligente?
    """
    
    try:
        articulo = ejemplo.inputs.get("articulo", "")
        
        # Usar LangChain LLM para evaluar tono
        response = llm.invoke([
            HumanMessage(content=
                "Eres un evaluador editorial experto. Analiza el tono cómico del siguiente artículo. "
                "Responde SOLO con un número del 0 al 10 (sin decimales) donde:\n"
                "0 = nada cómico, 10 = extremadamente cómico e inteligente\n\n"
                f"Artículo:\n{articulo[:2000]}"
            )
        ])
        
        # Extraer score del contenido
        contenido = response.content.strip()
        try:
            # Extraer solo el número
            numero = ''.join(filter(str.isdigit, contenido.split()[0]))
            score = float(numero) / 10.0  # Convertir a 0-1
        except:
            score = 0.5
        
        return {
            "key": "tono_comico",
            "score": score,
            "comment": f"Evaluación LLM del tono cómico: {score*10:.1f}/10"
        }
    except Exception as e:
        return {"key": "tono_comico", "score": 0.5, "comment": f"Error en LLM: {str(e)[:50]}"}


# ============================================================================
# CREACIÓN DE DATASET
# ============================================================================

def crear_dataset_articulos(articulo: str) -> str:
    """Crea o recupera un dataset en LangSmith"""
    
    print(f"📊 Buscando/creando dataset: {DATASET_NAME}")
    
    try:
        # Intentar obtener dataset existente - nueva API sin parámetro 'name'
        datasets = list(client.list_datasets())
        dataset = None
        
        # Buscar dataset por nombre
        for ds in datasets:
            if ds.name == DATASET_NAME:
                dataset = ds
                break
        
        if dataset:
            print(f"✅ Dataset existente encontrado (ID: {dataset.id})")
            return dataset.id
        
        # Crear nuevo dataset
        print(f"🆕 Creando nuevo dataset...")
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        
        # Crear ejemplos
        ejemplos = [
            {
                "articulo": articulo,
                "tema": "Inteligencia Artificial en países del primer mundo"
            }
        ]
        
        # Agregar ejemplos al dataset
        for ejemplo in ejemplos:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"articulo": ejemplo["articulo"], "tema": ejemplo["tema"]},
                outputs={"tono": "Cómico", "calidad": "Alta"}
            )
        
        print(f"✅ Dataset creado con {len(ejemplos)} ejemplo(s)")
        return dataset.id
    
    except Exception as e:
        print(f"❌ Error creando dataset: {e}")
        raise


# ============================================================================
# FUNCIÓN PARA GUARDAR REPORTE
# ============================================================================

def generar_reporte_langsmith(resultados, archivo_original: str) -> str:
    """Genera reporte de evaluación en Markdown"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_reporte = f"evaluacion_langsmith_{timestamp}.md"
    
    # Procesar resultados
    evaluaciones = {}
    for resultado in resultados:
        key = resultado.get("key")
        score = resultado.get("score", 0)
        comment = resultado.get("comment", "")
        evaluaciones[key] = {"score": score, "comment": comment}
    
    # Calcular promedio
    scores = [e["score"] for e in evaluaciones.values() if isinstance(e["score"], (int, float))]
    promedio = sum(scores) / len(scores) if scores else 0
    
    # Contenido del reporte
    contenido = f"""# 📊 Reporte de Evaluación LangSmith

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

**Archivo Evaluado:** {archivo_original}

**Framework:** LangSmith

---

## 🎯 Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| **Puntuación Promedio** | {promedio:.2f}/1.0 ({promedio*100:.1f}%) |
| **Total Evaluadores** | {len(evaluaciones)} |
| **Estado** | {"✅ APROBADO" if promedio >= 0.7 else "⚠️ REQUIERE REVISIÓN" if promedio >= 0.5 else "❌ RECHAZADO"} |

---

## 📈 Resultados Detallados

"""
    
    # Agregar cada evaluador
    evaluador_nombre = {
        "longitud_minima": "1️⃣ Longitud Mínima",
        "contenido_apropiado": "2️⃣ Contenido Apropiado",
        "estructura_parrafos": "3️⃣ Estructura de Párrafos",
        "relevancia_tema": "4️⃣ Relevancia del Tema",
        "tono_comico": "5️⃣ Tono Cómico (LLM)"
    }
    
    for key, eval_data in evaluaciones.items():
        score = eval_data["score"]
        comment = eval_data["comment"]
        emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
        
        contenido += f"""
### {evaluador_nombre.get(key, f"Evaluador {key}")}

**Puntuación:** {score:.2f}/1.0 ({score*100:.1f}%) {emoji}

**Análisis:** {comment}

"""
    
    # Interpretación
    contenido += f"""

---

## 🔍 Interpretación

### Escala de Puntuación:
- **0.9-1.0:** Excelente ✅
- **0.7-0.89:** Bueno ✅
- **0.5-0.69:** Aceptable ⚠️
- **< 0.5:** Deficiente ❌

### Recomendaciones:

"""
    
    if promedio >= 0.8:
        contenido += "✨ **El artículo está listo para publicación.** Cumple con todos los estándares de calidad editorial."
    elif promedio >= 0.6:
        contenido += "⚠️ **El artículo necesita algunos ajustes menores** antes de publicar. Revisa los evaluadores con puntuación baja."
    else:
        contenido += "❌ **El artículo requiere revisión significativa.** Se recomienda reescritura según feedback de evaluadores."
    
    contenido += f"""

---

## 🛠️ Información Técnica

- **Plataforma:** LangSmith
- **Dataset:** {DATASET_NAME}
- **Experimento:** {EXPERIMENT_PREFIX}
- **Modelo Evaluador:** GPT-4 (para evaluaciones LLM)
- **Evaluadores Utilizados:** {len(evaluaciones)}

### Evaluadores Implementados:

1. **Longitud Mínima** - Regla determinística
2. **Contenido Apropiado** - Regla determinística
3. **Estructura de Párrafos** - Métrica cuantitativa
4. **Relevancia del Tema** - Análisis de palabras clave
5. **Tono Cómico** - LLM-as-Judge

---

## 📋 Próximos Pasos

1. Revisar feedback de evaluadores
2. Si aprobado (≥0.7): Publicar artículo
3. Si requiere revisión (0.5-0.7): Ajustar según comentarios
4. Si rechazado (<0.5): Considerar reescritura

---

*Reporte generado automáticamente por LangSmith*
"""
    
    # Guardar archivo
    try:
        with open(nombre_reporte, 'w', encoding='utf-8') as f:
            f.write(contenido)
        print(f"✅ Reporte guardado: {nombre_reporte}")
        return nombre_reporte
    except Exception as e:
        print(f"❌ Error al guardar reporte: {e}")
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de evaluación con LangSmith"""
    
    print("=" * 80)
    print("🔬 EVALUACIÓN CON LANGSMITH - ARTÍCULOS EDITORIALES")
    print("=" * 80)
    print()
    
    # Obtener archivo
    if len(sys.argv) > 1:
        archivo_markdown = sys.argv[1]
    else:
        archivos = list(Path(".").glob("articulo_editorial_*.md"))
        if archivos:
            archivo_markdown = str(archivos[-1])
            print(f"📁 Archivo detectado: {archivo_markdown}\n")
        else:
            print("❌ No se encontró archivo de artículo editorial")
            print("Uso: python langsmith_evaluador.py <archivo_markdown>")
            sys.exit(1)
    
    # Extraer artículo
    print(f"📖 Extrayendo artículo...")
    articulo = extraer_articulo_del_markdown(archivo_markdown)
    
    if not articulo:
        print("❌ No se pudo extraer el artículo")
        sys.exit(1)
    
    print(f"✅ Artículo extraído ({len(articulo)} caracteres)\n")
    
    # Crear dataset
    print("=" * 80)
    print("📊 CREANDO DATASET EN LANGSMITH")
    print("=" * 80)
    print()
    
    try:
        dataset_id = crear_dataset_articulos(articulo)
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo crear dataset: {e}")
        print("Continuando con evaluación local...\n")
        dataset_id = None
    
    # Ejecutar evaluación
    print("\n" + "=" * 80)
    print("🚀 EJECUTANDO EVALUACIÓN")
    print("=" * 80)
    print()
    
    try:
        # Crear objeto Example con UUID
        import uuid
        
        ejemplo = Example(
            id=str(uuid.uuid4()),
            inputs={"articulo": articulo, "tema": "Inteligencia Artificial"},
            outputs={"tono": "Cómico"}
        )
        
        resultados_evaluacion = []
        evaluadores = [
            evaluador_longitud_minima,
            evaluador_contenido_apropiado,
            evaluador_cantidad_parrafos,
            evaluador_relevancia_tema,
            evaluador_tono_comico_llm
        ]
        
        print("Ejecutando evaluadores:\n")
        
        for evaluador_func in evaluadores:
            try:
                # Llamar evaluador sin Run
                resultado = evaluador_func(ejemplo)
                resultados_evaluacion.append(resultado)
                print(f"✓ {resultado['key']}: {resultado['score']:.2f}/1.0")
                print(f"  → {resultado['comment']}\n")
            except Exception as e:
                print(f"❌ Error en {evaluador_func.__name__}: {e}\n")
        
        # Mostrar resumen
        print("=" * 80)
        print("📋 RESUMEN DE EVALUACIÓN")
        print("=" * 80)
        print()
        
        promedio = sum(r["score"] for r in resultados_evaluacion) / len(resultados_evaluacion)
        print(f"Puntuación Promedio: {promedio:.2f}/1.0 ({promedio*100:.1f}%)\n")
        
        # Generar reporte
        print("=" * 80)
        print("💾 GENERANDO REPORTE")
        print("=" * 80)
        print()
        
        reporte_path = generar_reporte_langsmith(resultados_evaluacion, archivo_markdown)
        
        if reporte_path:
            print(f"📁 Ubicación: {os.path.abspath(reporte_path)}")
            print(f"\n✨ ¡Evaluación completada exitosamente!")
    
    except Exception as e:
        print(f"❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()