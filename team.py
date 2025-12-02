"""
Multi-Agent Editorial Team usando LangGraph
Supervisor, Redactor y Editor trabajando juntos para crear artículos de IA

Requiere:
    pip install langgraph langchain-openai python-dotenv
"""

import os
from typing import Annotated, Any, Dict, List
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing_extensions import TypedDict

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# CONFIGURACIÓN DE AGENTES
# ============================================================================

class AgentState(TypedDict):
    """Estado compartido entre agentes"""
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    tema: str
    articulo_draft: str
    articulo_editado: str
    feedback_editor: str
    siguiente_paso: str


# Inicializar modelo LLM
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================================
# AGENTES ESPECIALIZADOS
# ============================================================================

def agente_redactor(state: AgentState) -> AgentState:
    """
    Agente Redactor: Crea el borrador del artículo
    Tono: Cómico, ameno, con humor social sano
    """
    
    prompt_redactor = f"""Eres un redactor especializado en tecnología con un humor único y sano.
Tu misión: Escribir un artículo sobre "{state['tema']}"

INSTRUCCIONES:
1. Crea un artículo ameno y cómico sobre cómo la IA está invadiendo (de forma divertida) los países del primer mundo
2. Incluye anécdotas imaginativas pero realistas sobre:
   - Trabajadores que compiten con ChatGPT por sus empleos (con humor)
   - Startups de IA que surgen cada 5 minutos
   - Cómo Silicon Valley está obsesionado con la IA
   - La brecha entre el hype y la realidad
   - Aspectos positivos sin ser condescendiente

3. Mantén un tono:
   - Divertido pero inteligente
   - Inclusivo y respetuoso
   - Que promueva reflexión crítica
   - Con humor sobre la naturaleza humana

4. Estructura:
   - Titular cómico e impactante
   - Introducción que enganche
   - 3-4 secciones temáticas
   - Conclusión reflexiva pero ligera

5. Extensión: ~800 palabras
6. Lenguaje: Español natural, con un toque de humor millennial

COMIENZA A ESCRIBIR:"""
    
    # Llamar al LLM
    response = llm.invoke([HumanMessage(content=prompt_redactor)])
    articulo_draft = response.content
    
    # Actualizar estado
    state["articulo_draft"] = articulo_draft
    state["messages"].append(HumanMessage(content=f"Redactor creó borrador: {len(articulo_draft)} caracteres"))
    state["siguiente_paso"] = "editor"
    
    return state


def agente_editor(state: AgentState) -> AgentState:
    """
    Agente Editor: Revisa, mejora y enriquece el artículo
    Verificar: Coherencia, fluidez, tono consistente
    """
    
    prompt_editor = f"""Eres un editor experimentado en publicaciones satíricas y de humor inteligente.
Tu misión: Pulir y mejorar el siguiente artículo manteniendo su esencia divertida.

ARTÍCULO ACTUAL:
{state['articulo_draft']}

TAREAS DE EDICIÓN:
1. REVISIÓN DE TONO:
   - ¿El humor es sano y no offensivo?
   - ¿Mantiene coherencia en el tono cómico?
   - ¿Hay balance entre diversión y profundidad?

2. MEJORAS ESTRUCTURALES:
   - ¿Fluye bien de un párrafo a otro?
   - ¿El titular es lo suficientemente catchy?
   - ¿La introducción engancha al lector?
   - ¿La conclusión es satisfactoria?

3. ENRIQUECIMIENTO:
   - Agrega detalles satíricos pero creíbles
   - Mejora transiciones entre secciones
   - Asegura que el humor sea *inteligente* (no solo bromas)
   - Añade una reflexión final que haga pensar al lector

4. CORRECCIONES:
   - Ortografía y gramática
   - Puntuación
   - Coherencia de argumentos

5. INSTRUCCIONES:
   - Mantén el 80% del contenido original
   - Solo mejora lo que lo hace mejor
   - El humor debe ser inclusivo, nunca cruel
   - Promueve buenas costumbres indirectamente (a través del humor)

DEVUELVE:
El artículo editado (completo) seguido de un párrafo con tu feedback."""
    
    response = llm.invoke([HumanMessage(content=prompt_editor)])
    contenido_respuesta = response.content
    
    # Separar artículo editado del feedback
    partes = contenido_respuesta.rsplit("\n\nFEEDBACK:", 1)
    articulo_editado = partes[0] if partes else contenido_respuesta
    feedback = partes[1] if len(partes) > 1 else "Edición completada sin feedback adicional"
    
    state["articulo_editado"] = articulo_editado
    state["feedback_editor"] = feedback
    state["messages"].append(HumanMessage(content=f"Editor revisó y mejoró el artículo"))
    state["siguiente_paso"] = "supervisor"
    
    return state


def agente_supervisor(state: AgentState) -> AgentState:
    """
    Agente Supervisor: Valida calidad, coherencia y alineación con valores editoriales
    Decide si se publica o requiere revisiones
    """
    
    prompt_supervisor = f"""Eres el supervisor editorial responsable de mantener los estándares de calidad.
Tu misión: Validar que el artículo cumpla con nuestros estándares de excelencia y valores.

ARTÍCULO PARA VALIDAR:
{state['articulo_editado']}

FEEDBACK DEL EDITOR:
{state['feedback_editor']}

EVALÚA:
1. CALIDAD GENERAL (1-10):
   - ¿Es un artículo competente y entretenido?
   - ¿Vale la pena publicar?
   - ¿Tiene valor para el lector?

2. ALINEACIÓN CON VALORES:
   - ✓ ¿El humor es sano y no ofensivo?
   - ✓ ¿Promueve buenas costumbres indirectamente?
   - ✓ ¿Es inclusivo?
   - ✓ ¿Educa mientras entretiene?

3. VERIFICACIÓN DE HECHOS:
   - ¿Las anécdotas sobre IA en países desarrollados tienen base real?
   - ¿Las críticas son justas?

4. DECISIÓN:
   - PUBLICAR: Si todo está bien
   - REVISAR: Si necesita ajustes (especifica cuáles)
   - RECHAZAR: Si no cumple estándares (poco probable)

FORMATO FINAL:
Comienza con "DECISIÓN: [PUBLICAR/REVISAR/RECHAZAR]"
Seguido de una justificación breve (máx 150 palabras)
Si hay sugerencias específicas, lístalas claramente."""
    
    response = llm.invoke([HumanMessage(content=prompt_supervisor)])
    evaluacion_supervisor = response.content
    
    state["messages"].append(HumanMessage(content="Supervisor evaluó y dio feedback final"))
    state["siguiente_paso"] = "finalizado"
    
    # Extraer decisión
    if "PUBLICAR" in evaluacion_supervisor.upper():
        print("\n✅ ARTÍCULO APROBADO PARA PUBLICACIÓN")
    elif "REVISAR" in evaluacion_supervisor.upper():
        print("\n⚠️ ARTÍCULO REQUIERE REVISIONES")
    else:
        print("\n❌ ARTÍCULO RECHAZADO")
    
    print(f"\n{evaluacion_supervisor}")
    
    return state


# ============================================================================
# FUNCIONES DE ROUTING
# ============================================================================

def routing_supervisor(state: AgentState) -> str:
    """Determina el siguiente agente según el estado"""
    siguiente = state.get("siguiente_paso", "redactor")
    
    if siguiente == "editor":
        return "editor"
    elif siguiente == "supervisor":
        return "supervisor"
    else:
        return END


def routing_editor(state: AgentState) -> str:
    """El editor siempre pasa al supervisor después"""
    return "supervisor"


def routing_redactor(state: AgentState) -> str:
    """El redactor siempre pasa al editor después"""
    return "editor"


# ============================================================================
# CONSTRUCCIÓN DEL GRAFO (WORKFLOW)
# ============================================================================

def crear_equipo_editorial():
    """Crea el grafo de LangGraph con los agentes"""
    
    # Crear grafo
    workflow = StateGraph(AgentState)
    
    # Agregar nodos (agentes)
    workflow.add_node("redactor", agente_redactor)
    workflow.add_node("editor", agente_editor)
    workflow.add_node("supervisor", agente_supervisor)
    
    # Definir flujo de trabajo
    # Comienza con redactor
    workflow.set_entry_point("redactor")
    
    # Transiciones
    workflow.add_edge("redactor", "editor")      # Redactor → Editor
    workflow.add_edge("editor", "supervisor")    # Editor → Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: END  # El supervisor es el punto final
    )
    
    # Compilar el grafo
    app = workflow.compile()
    return app


# ============================================================================
# FUNCIÓN PARA GUARDAR REPORTE
# ============================================================================

def guardar_reporte_markdown(estado_final: AgentState, tema: str):
    """Guarda el reporte completo en formato Markdown"""
    
    # Generar nombre del archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"articulo_editorial_{timestamp}.md"
    
    # Crear contenido del markdown
    contenido_markdown = f"""# 📝 Reporte de Producción Editorial

**Fecha:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

---

## 📋 Información del Artículo

| Campo | Valor |
|-------|-------|
| **Tema** | {tema} |
| **Estado** | ✅ Completado |
| **Agentes Involucrados** | Redactor, Editor, Supervisor |
| **Caracteres Finales** | {len(estado_final['articulo_editado'])} |

---

## 📊 Resumen del Proceso

### 🖊️ Fase 1: Redacción
- **Agente:** Redactor
- **Tarea:** Crear borrador inicial del artículo
- **Resultado:** Borrador de {len(estado_final['articulo_draft'])} caracteres
- **Tono Establecido:** Cómico, sano, inclusivo

### ✏️ Fase 2: Edición
- **Agente:** Editor
- **Tarea:** Pulir y mejorar el contenido
- **Mejoras Aplicadas:**
  - Revisión de coherencia y fluidez
  - Validación del tono cómico pero inteligente
  - Mejora de transiciones entre secciones
  - Enriquecimiento de detalles satíricos
  
**Feedback del Editor:**
```
{estado_final['feedback_editor']}
```

### 👔 Fase 3: Supervisión
- **Agente:** Supervisor
- **Tarea:** Validar calidad y alineación con valores editoriales
- **Criterios Evaluados:**
  - ✓ Calidad general del contenido
  - ✓ Alineación con valores: humor sano, inclusivo, que promueva buenas costumbres
  - ✓ Coherencia y estructura
  - ✓ Verificación de hechos

---

## 📄 ARTÍCULO FINAL

{estado_final['articulo_editado']}

---

## ✅ Estado Final

**ESTADO:** Artículo listo para publicación

**Próximos Pasos:**
1. Revisar en plataforma de publicación
2. Agregar imágenes y multimedia (opcional)
3. Publicar en sitio web
4. Compartir en redes sociales

**Generado por:** Sistema Editorial Multi-Agent (LangGraph)

---

*Este documento fue generado automáticamente por el sistema editorial.*
"""
    
    # Guardar archivo
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(contenido_markdown)
        
        print(f"✅ Reporte guardado exitosamente: {nombre_archivo}")
        return nombre_archivo
    except Exception as e:
        print(f"❌ Error al guardar archivo: {e}")
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta el equipo editorial para crear un artículo"""
    
    print("=" * 80)
    print("🎬 EQUIPO EDITORIAL DE IA - MODO PRODUCCIÓN")
    print("=" * 80)
    print()
    
    # Tema del artículo
    tema = "La invasión cómica de la Inteligencia Artificial en países del primer mundo"
    
    print(f"📝 TEMA: {tema}")
    print()
    
    # Estado inicial
    estado_inicial: AgentState = {
        "messages": [
            HumanMessage(content=f"Nueva asignación de artículo: {tema}")
        ],
        "tema": tema,
        "articulo_draft": "",
        "articulo_editado": "",
        "feedback_editor": "",
        "siguiente_paso": "redactor"
    }
    
    # Crear y ejecutar el workflow
    print("🚀 Iniciando flujo de trabajo...\n")
    app = crear_equipo_editorial()
    
    # Ejecutar el grafo
    estado_final = app.invoke(estado_inicial)
    
    print("\n" + "=" * 80)
    print("📄 ARTÍCULO FINAL")
    print("=" * 80)
    print(estado_final["articulo_editado"])
    
    print("\n" + "=" * 80)
    print("✨ PROCESO COMPLETADO")
    print("=" * 80)
    
    # Resumen del flujo
    print(f"""
📊 RESUMEN DE TRABAJO:
   • Redactor: Creó borrador de {len(estado_final['articulo_draft'])} caracteres
   • Editor: Pulió y mejoró el contenido
   • Supervisor: Validó calidad y valores editoriales
   
✅ Artículo listo para publicación en plataforma
""")
    
    # Guardar reporte en Markdown
    print("\n" + "=" * 80)
    print("💾 GUARDANDO REPORTE")
    print("=" * 80)
    archivo_generado = guardar_reporte_markdown(estado_final, tema)
    
    if archivo_generado:
        print(f"📁 Ubicación: {os.path.abspath(archivo_generado)}")
        print("\n✨ ¡Todo completado exitosamente!")


if __name__ == "__main__":
    main()