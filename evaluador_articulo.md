## 📊 **Características del Evaluador:**

### **5 Métricas Personalizadas:**

1. **Tono Cómico y Ameno** 🎭
   - Mide humor inteligente, sarcasmo constructivo
   - Evita ofensas

2. **Conformidad con Valores Editoriales** 💚
   - Promueve buenas costumbres
   - Inclusivo y respetuoso
   - Educativo

3. **Estructura y Coherencia** 📐
   - Titular, introducción, secciones, conclusión
   - Transiciones fluidas

4. **Relevancia y Precisión sobre IA** 🤖
   - Contenido actual y creíble
   - Balance entre hype y realidad
   - Ejemplos realistas

5. **Engagement y Atracción** ⭐
   - ¿Mantiene atención del lector?
   - ¿Es página-turner?

## 🚀 **Uso:**

```bash
# Opción 1: Especificar archivo
python evaluador_articulo.py articulo_editorial_20250101_143022.md

# Opción 2: Auto-detecta el archivo más reciente
python evaluador_articulo.py
```

## 📊 Total de Métricas: 12

   -  Built-in de Confident AI (Seguridad, Fidelidad, Relevancia)
   -  Personalizadas con GEval (3 de Tono + Corrección IA + Engagement)

## 📁 **Salida:**

✅ **Consola**: Resultados inmediatos con emojis
✅ **Archivo Markdown**: `evaluacion_articulo_YYYYMMDD_HHMMSS.md`

## 📋 **Contenido del Reporte:**

- Puntuación promedio general
- Detalles de cada métrica con razonamiento
- Interpretación de resultados
- Recomendaciones personalizadas
- Próximos pasos según puntuación

## 📋 Categorías en el Reporte:

Métricas de Seguridad (Bias & Toxicity)
Métricas de Fidelidad (Hallucination & Faithfulness)
Métricas de Relevancia (RAG & Content)
Métricas de Tono Cómico ✨ (Nuevo)
Otras métricas editoriales

## 🎯 **Escala de Calificación:**

- **9-10**: Excelente ✅ → Publicar
- **7-8**: Bueno ✅ → Publicar
- **5-6**: Aceptable ⚠️ → Revisar
- **0-4**: Deficiente ❌ → Rechazar

El evaluador **usa LLM-as-Judge** (GPT-4) para evaluar el artículo con criterios editoriales específicos. ¡Listo para evaluar tus artículos! 🚀