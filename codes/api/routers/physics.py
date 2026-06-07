from fastapi import APIRouter, HTTPException
from ..models import FRFRequest
import sys
import os

# Ensure codes directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.FRF import frf

router = APIRouter(prefix="/physics", tags=["Physics"])

@router.post("/calculate-frf")
async def calculate_frf(request: FRFRequest):
    """
    Compute the Frequency Response Function (FRF) for a given DVA configuration.
    """
    try:
        # Construct DVA parameters list as expected by FRF module (48 parameters)
        # Format: (mu1, mu2, mu3, lambda1-15, nu1-15, beta1-15)
        dva_params = [
            request.dva_params.mu_1,
            request.dva_params.mu_2,
            request.dva_params.mu_3,
            *request.dva_params.lambda_1_15,
            *request.dva_params.nu_1_15,
            *request.dva_params.beta_1_15
        ]
        
        # Call the core FRF solver with all required positional arguments
        empty_dict = {}
        results = frf(
            main_system_parameters=request.main_system_params,
            dva_parameters=dva_params,
            omega_start=request.omega_range[0],
            omega_end=request.omega_range[1],
            omega_points=int(request.omega_range[2]),
            target_values_mass1=empty_dict,
            weights_mass1=empty_dict,
            target_values_mass2=empty_dict,
            weights_mass2=empty_dict,
            target_values_mass3=empty_dict,
            weights_mass3=empty_dict,
            target_values_mass4=empty_dict,
            weights_mass4=empty_dict,
            target_values_mass5=empty_dict,
            weights_mass5=empty_dict,
            plot_figure=False,
            interpolation_method=request.interpolation_method
        )
        
        # Process results into JSON serializable format
        # results typically contains keys like 'mass_1', 'mass_2', and 'singular_response'
        serializable_results = {}
        for key, data in results.items():
            if key.startswith("mass_"):
                mass_idx = int(key.split("_")[1])
                if mass_idx in request.target_masses:
                    serializable_results[str(mass_idx)] = {
                        "peak_positions": data.get("peak_positions", {}),
                        "peak_values": data.get("peak_values", {}),
                        "singular_response": float(data.get("singular_response", 0.0))
                    }
            elif key == "singular_response":
                serializable_results["total_singular_response"] = float(data)
        
        return {
            "status": "success",
            "results": serializable_results
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
