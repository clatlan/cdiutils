"""
Test that pipeline imports work correctly without requiring optional
dependencies.

This ensures that notebooks can import cdiutils.pipeline without needing
PyNX or ipywidgets installed.
"""


def test_pipeline_import_without_gui_dependencies():
    """Test that pipeline can be imported without PyNX/ipywidgets."""
    import cdiutils

    # This is the pattern used in notebooks
    params_func = cdiutils.pipeline.get_params_from_variables
    assert params_func is not None

    # BcdiPipeline should be accessible
    bcdi_cls = cdiutils.BcdiPipeline
    assert bcdi_cls is not None


def test_bcdi_pipeline_class_import():
    """Test direct import of BcdiPipeline."""
    from cdiutils.pipeline import BcdiPipeline

    assert BcdiPipeline is not None


def test_phase_retrieval_gui_lazy_import():
    """Test that PhaseRetrievalGUI import is lazy and doesn't break
    BcdiPipeline.
    """
    from cdiutils.pipeline import BcdiPipeline

    # BcdiPipeline should import successfully
    assert hasattr(BcdiPipeline, "phase_retrieval_gui")

    # The GUI method should exist but not import PhaseRetrievalGUI until called
    # (we can't actually call it without dependencies, but we can verify it
    # exists)
    assert callable(BcdiPipeline.phase_retrieval_gui)


def test_get_params_from_variables():
    """Test get_params_from_variables function."""
    from cdiutils.pipeline import get_params_from_variables

    # Test with empty variables
    params = get_params_from_variables([], {})
    assert isinstance(params, dict)
