## 🚀 **Características Principales:**

### **5 Evaluadores Personalizados:**

1. **Longitud Mínima** (Regla)
   - ¿Artículo tiene ≥500 caracteres?

2. **Tono Cómico** (LLM-as-Judge)
   - Usa GPT-4 para evaluar humor inteligente

3. **Contenido Apropiado** (Regla)
   - Detecta contenido inapropiado
   - Verifica que sea sano e inclusivo

4. **Estructura de Párrafos** (Métrica)
   - ¿Tiene ≥4 párrafos bien organizados?

5. **Relevancia del Tema** (Análisis de palabras clave)
   - Verifica menciones de IA, tecnología, países desarrollados

### **Flujo Completo:**

```
Extrae Artículo → Crea Dataset → Ejecuta Evaluadores → Genera Reporte
```

## 📊 **Configuración Requerida:**

```bash
# Instalar dependencias
pip install langsmith langchain-openai python-dotenv

# Configurar variables de entorno
export LANGSMITH_API_KEY="tu-api-key"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export OPENAI_API_KEY="tu-openai-key"
```

## 🎯 **Uso:**

```bash
# Especificar archivo
python langsmith_evaluador.py articulo_editorial_20250101_143022.md

# O auto-detectar
python langsmith_evaluador.py
```

## 📁 **Salida:**

✅ **Consola**: Resultados inmediatos de cada evaluador
✅ **LangSmith Dashboard**: Experiment tracking en la plataforma
✅ **Archivo Markdown**: `evaluacion_langsmith_YYYYMMDD_HHMMSS.md`

## 📈 **Reporte Incluye:**

- Puntuación de cada evaluador (0-1.0)
- Promedio general
- Comentarios detallados
- Recomendaciones según puntuación
- Status: APROBADO / REQUIERE REVISIÓN / RECHAZADO

## 🔗 **Integración LangSmith:**

- Crea/usa dataset "Artículos Editoriales IA"
- Registra experimento en LangSmith
- Puedes ver resultados en el dashboard web
- Compara múltiples experimentos

¡Ahora tienes evaluación con **LangSmith** lista para usar! 🎊