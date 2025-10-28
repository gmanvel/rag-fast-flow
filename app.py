"""
Streamlit application for comparing pure LLM vs RAG-enhanced responses.
"""

import streamlit as st
from llm_service import LLMService
from rag_service import RAGService


# Page configuration
st.set_page_config(
    page_title="Fast Flow RAG Comparison",
    page_icon="🚀",
    layout="wide"
)

# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache LLM and RAG services."""
    llm_service = LLMService()
    rag_service = RAGService()
    return llm_service, rag_service


def main():
    """Main application function."""

    # Header
    st.title("🚀 Fast Flow Methodologies Q&A")
    st.markdown("""
    Compare responses from a **Pure LLM** vs. **RAG-Enhanced LLM** for questions about
    Fast Flow methodologies (Wardley Mapping, Domain-Driven Design, Team Topologies).
    """)

    # Initialize services
    try:
        llm_service, rag_service = get_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()

    # Sidebar with connection status
    with st.sidebar:
        st.header("⚙️ System Status")

        # Check Qdrant connection
        with st.spinner("Checking Qdrant connection..."):
            qdrant_status = rag_service.check_connection()

        if qdrant_status["status"] == "connected":
            if qdrant_status.get("collection_exists"):
                points_count = qdrant_status['points_count']
                if points_count > 0:
                    st.success("✅ Qdrant Connected")
                    st.info(f"📊 {points_count} chunks loaded")
                    st.info(f"🔢 Vector size: {qdrant_status['vector_size']}")
                else:
                    st.warning("⚠️ Qdrant collection is empty")
                    st.caption(f"Collection `{rag_service.collection_name}` has no data")
            else:
                st.warning("⚠️ Qdrant connected but collection not found")
                st.caption(f"Expected: `{rag_service.collection_name}`")
                if qdrant_status.get("available_collections"):
                    st.caption(f"Available: {', '.join(qdrant_status['available_collections'])}")

            # Show populate button if collection doesn't exist or is empty
            collection_exists = qdrant_status.get("collection_exists", False)
            points_count = qdrant_status.get("points_count", 0)
            show_populate_button = not collection_exists or points_count == 0

            if show_populate_button:
                st.divider()
                st.subheader("📥 Data Population")

                if st.button("🚀 Populate Qdrant", type="primary", use_container_width=True):
                    with st.spinner("Loading data from JSON..."):
                        st.caption("⏳ This may take a few minutes...")

                        # Call populate_database
                        result = rag_service.populate_database('/app/data/fast_flow_extracted.json')

                        if result.get("success"):
                            st.success("✅ Database populated successfully!")
                            st.info(f"📄 Processed {result['sections_processed']} sections")
                            st.info(f"🔢 Created {result['chunks_created']} chunks")
                            st.caption("Refreshing page...")

                            # Clear cache and refresh
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to populate database")
                            st.caption(f"Error: {result.get('error', 'Unknown error')}")
        else:
            st.error("❌ Qdrant connection failed")
            st.caption(qdrant_status.get("error", "Unknown error"))

        st.divider()

        st.header("ℹ️ About")
        st.markdown("""
        **Pure LLM**: Direct query to Mistral without additional context.

        **RAG-Enhanced**: Query enriched with relevant sections from the Fast Flow documentation retrieved from Qdrant.
        """)

        st.divider()

        st.header("🔧 Configuration")
        st.caption(f"Qdrant: {rag_service.qdrant_host}:{rag_service.qdrant_port}")
        st.caption(f"Ollama: {llm_service.base_url}")
        st.caption(f"Model: {llm_service.model}")

    # Main input area
    st.header("💬 Ask a Question")

    question = st.text_area(
        "Enter your question about Finding Software Boundaries for Fast Flow",
        placeholder="e.g., What is Doctrine in Wardley Maps?",
        height=100
    )

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        submit_button = st.button("🔍 Get Answers", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    # Process query
    if submit_button and question.strip():
        st.divider()

        # Create two columns for side-by-side comparison
        col_pure, col_rag = st.columns(2)

        with col_pure:
            st.subheader("💭 Pure LLM Response")
            with st.spinner("Generating pure LLM response..."):
                try:
                    pure_response = llm_service.query_pure(question)
                    st.markdown(pure_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col_rag:
            st.subheader("🎯 RAG-Enhanced Response")

            # First, retrieve context
            with st.spinner("Retrieving relevant context..."):
                try:
                    context = rag_service.retrieve_context(question)

                    # Show retrieved context in expander
                    with st.expander("📚 Retrieved Context", expanded=False):
                        st.text(context)

                    # Generate RAG response
                    with st.spinner("Generating RAG-enhanced response..."):
                        rag_response = llm_service.query_rag(question, context)
                        st.markdown(rag_response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif submit_button:
        st.warning("Please enter a question first.")

    # Footer
    st.divider()
    st.caption("""
    This application demonstrates the difference between pure LLM responses and RAG-enhanced responses.
    RAG (Retrieval-Augmented Generation) enriches the LLM's context with relevant information from a knowledge base.
    """)


if __name__ == "__main__":
    main()
