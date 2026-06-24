# CrystaLLM Crystal Structure Generation Module

## 📋 Overview

This module provides crystal structure generation using CrystaLLM, a large language model specifically trained for generating crystal structures from chemical compositions.

## 🎯 Features

- **Composition-to-Structure**: Generate CIF files from chemical formulas
- **Token-Optimized Output**: Returns file paths instead of full CIF content to reduce token consumption
- **Modular Design**: Clean separation of concerns
- **MCP Integration**: Exposed as MCP tool for agent use
- **CPU/CUDA Support**: Flexible device selection
- **Persistent Storage**: Generated CIF files are saved to `generated_structures/` directory for downstream use

## 📁 Module Structure

```
crystallm/
├── __init__.py           # Package initialization
├── generator.py          # Core generation logic
├── test_generator.py     # Test script
└── README.md            # This file
```

## 🚀 Usage

### As a Python Module

```python
from crystallm import generate_crystal_from_composition

# Generate crystal structure
result = generate_crystal_from_composition(
    composition="GaN",
    device="cpu",
    num_samples=1
)

if result['success']:
    # Result now returns file paths instead of full CIF content (optimized for token usage)
    cif_file_path = result['cif_file_paths'][0]
    cif_filename = result['cif_filenames'][0]
    print(f"Generated: {cif_filename}")
    print(f"Saved to: {cif_file_path}")

    # Read CIF content when needed
    with open(cif_file_path, 'r', encoding='utf-8') as f:
        cif_content = f.read()
    print(f"CIF content length: {len(cif_content)} characters")
else:
    print(f"Error: {result['error']}")
```

### As an MCP Tool

The module is exposed as an MCP tool in `mcp_servers/simulation/server.py`:

```python
# In your agent
result = await call_tool(
    "generate_crystal_structure",
    composition="GaN",
    device="cpu",
    num_samples=1
)
```

### Using the CrystaLLM Agent

```bash
# Interactive mode
python -m agents.crystallm_agent.agent

# Generate specific composition
python -m agents.crystallm_agent.agent GaN
```

## 📊 Function Reference

### `generate_crystal_from_composition`

Generate crystal structure from chemical composition.

**Parameters**:
- `composition` (str): Chemical composition (e.g., "Si", "GaN", "Fe2O3", "LiFePO4")
- `device` (str): Computing device - "cpu" (default) or "cuda"
- `num_samples` (int): Number of structures to generate (default: 1)
- `top_k` (int): Top-k sampling parameter (default: 10)
- `max_new_tokens` (int): Maximum tokens to generate (default: 2000)

**Returns**:
```python
{
    "success": bool,              # Whether generation succeeded
    "cif_file_paths": List[str],  # Paths to generated CIF files (optimized for token usage)
    "cif_filenames": List[str],   # Generated CIF filenames
    "cif_directory": str,         # Directory containing all generated CIF files
    "composition": str,           # Input composition
    "generation_id": str,         # Unique generation ID
    "num_generated": int,         # Number of structures generated
    "cif_source": str,            # "postprocessed" or "raw"
    "model_used": str,            # Model name used
    "device": str,                # Device used for generation
    "frontend_structures": List[Dict],  # Frontend-compatible structure data (includes cifContent)
    "num_frontend_structures": int,     # Number of frontend structures
    "error": str                  # Error message if failed (only if success=False)
}
```

## 🔬 Examples

### Example 1: Simple Generation

```python
result = generate_crystal_from_composition("Si")

if result['success']:
    print(f"✅ Generated {result['cif_filenames'][0]}")
    print(f"📁 Saved to: {result['cif_file_paths'][0]}")
    print(f"Model: {result['model_used']}")
    print(f"Device: {result['device']}")
```

### Example 2: Multiple Samples

```python
result = generate_crystal_from_composition(
    composition="GaN",
    device="cpu",
    num_samples=5
)

if result['success']:
    print(f"Generated {result['num_generated']} structures")
    print(f"📁 Directory: {result['cif_directory']}")
    for i, path in enumerate(result['cif_file_paths']):
        print(f"  {i+1}. {result['cif_filenames'][i]} -> {path}")
```

### Example 3: With Property Calculation

```python
# Generate structure
gen_result = generate_crystal_from_composition("GaN")

if gen_result['success']:
    # Read CIF content from file path (optimized for token usage)
    cif_path = gen_result['cif_file_paths'][0]
    with open(cif_path, 'r', encoding='utf-8') as f:
        cif_content = f.read()

    # Calculate thermal conductivity
    from modules import calculate_kappa_from_cif_impl

    kappa_result = calculate_kappa_from_cif_impl(
        cif_content=cif_content,
        cif_filename=gen_result['cif_filenames'][0],
        method="kappa_p"
    )

    if kappa_result['success']:
        print(f"Thermal Conductivity: {kappa_result['kappa']} W/(m·K)")
```

## 🧪 Testing

Run the test script:

```bash
cd mcp_servers/simulation/crystallm
python test_generator.py
```

Expected output:
```
================================================================================
CrystaLLM Crystal Structure Generation Test
================================================================================

📋 Test Parameters:
   Composition: Si
   Device: cpu
   Number of samples: 1

🔧 Generating crystal structure...

================================================================================
📊 Results
================================================================================

✅ Generation Successful!

📄 CIF Filename: Si_abc123.cif
🔬 Composition: Si
🆔 Generation ID: abc123
📦 Number Generated: 1
🤖 Model Used: crystallm_v1_large
💻 Device: cpu

📝 CIF Content (first 30 lines):
--------------------------------------------------------------------------------
  1: data_Si
  2: _cell_length_a    5.431
  3: _cell_length_b    5.431
  ...
--------------------------------------------------------------------------------

💾 Saved to: test_Si_abc123.cif

================================================================================
✅ Test Completed Successfully!
================================================================================
```

## 🛠️ Technical Details

### Dependencies

- **CrystaLLM**: Crystal structure generation model
- **PyMatGen**: Crystal structure analysis
- **PyTorch**: Deep learning framework

### Model Location

The module looks for CrystaLLM models in:
1. `D:/llmkappa/CrystaLLM/pre-trained-model/crystallm_v1_large`
2. `D:/llmkappa/CrystaLLM/pre-trained-model/crystallm_v1_small`
3. `CrystaLLM/pre-trained-model/crystallm_v1_large`
4. `CrystaLLM/pre-trained-model/crystallm_v1_small`

### Generated Files

The module creates persistent directories in `generated_structures/` for:
- Prompts: `{composition}_{generation_id}/prompts/`
- Generated structures: `{composition}_{generation_id}/generated/`
- Processed structures: `{composition}_{generation_id}/processed/`

**Important**: Generated CIF files are kept in the `generated_structures/` directory for downstream use (thermal conductivity, phonon calculations, etc.). This enables:
1. Token-efficient responses (file paths instead of full content)
2. Direct file access for downstream tools
3. Reproducibility and traceability of generated structures

## 🔧 Configuration

### Device Selection

**CPU (Recommended)**:
- More stable
- No GPU memory issues
- Sufficient for most cases

```python
result = generate_crystal_from_composition("GaN", device="cpu")
```

**CUDA (Optional)**:
- Faster generation
- Requires GPU with sufficient memory

```python
result = generate_crystal_from_composition("GaN", device="cuda")
```

### Number of Samples

**Single Sample (Recommended)**:
```python
result = generate_crystal_from_composition("GaN", num_samples=1)
```

**Multiple Samples**:
```python
result = generate_crystal_from_composition("GaN", num_samples=5)
```

Note: Only the first generated structure is returned in the result.

## 📚 Integration with Agents

### CrystaLLM Agent

A dedicated agent is provided in `agents/crystallm_agent/`:

**Features**:
- Natural language interaction
- Automatic structure generation
- Property calculation integration
- Friendly user communication

**Usage**:
```python
from agents.crystallm_agent import run_crystallm_agent

# Interactive mode
run_crystallm_agent()

# Direct generation
run_crystallm_agent(composition="GaN")
```

### Agent Workflow

1. User provides composition
2. Agent calls `generate_crystal_structure` tool
3. Agent presents CIF content
4. Agent offers property calculations
5. User can request thermal conductivity or energy properties
6. Agent calls appropriate calculation tools

## 🐛 Troubleshooting

### Issue: "CrystaLLM not found"

**Solution**: Check that CrystaLLM is installed at `D:/llmkappa/CrystaLLM`

### Issue: "Model not found"

**Solution**: Ensure pre-trained models are in `CrystaLLM/pre-trained-model/`

### Issue: "No CIF files generated"

**Possible causes**:
- Invalid composition
- Model generation failed
- Insufficient memory

**Solution**: Check logs, try simpler composition, use CPU device

### Issue: "Import error"

**Solution**: Ensure CrystaLLM is properly installed and Python path is set

## 📖 References

- **CrystaLLM**: Crystal structure generation using large language models
- **Reference Code**: `D:\llmkappa\bin\crystal_generator.py`
- **Workflow Config**: `D:\llmkappa\workflow-single.py`

## 🎉 Summary

This module provides a clean, modular interface to CrystaLLM for crystal structure generation:

✅ Simple API
✅ MCP tool integration
✅ Agent support
✅ Automatic cleanup
✅ Comprehensive error handling
✅ CPU/CUDA support

Ready to generate crystal structures from chemical compositions!

