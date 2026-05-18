# Internal
from src.models.vit.vit import STViT
# External


def count_parameters_per_module(model):
    print(f"{'Module Name':<25} | {'Parameters':<15}")
    print("-" * 45)
    
    total_params = 0
    for name, child in model.named_children():
        # Sum parameters for this specific child module
        module_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"{name:<25} | {module_params:<15,}")
        total_params += module_params
        
    # Don't forget standalone nn.Parameters (like your tags)
    # These are in model.parameters() but not necessarily in named_children()
    all_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    standalone_params = all_params - total_params
    
    if standalone_params > 0:
        print(f"{'Standalone Parameters':<25} | {standalone_params:<15,}")
    
    print("-" * 45)
    print(f"{'TOTAL':<25} | {all_params:<15,}")



if __name__ == '__main__':
    model = STViT(
        num_static_channels=6,
        num_dynamic_channels=7,
        num_timestamps_per_sample=10,
        patch_size=8,
        embedding_dim=128
    )

    count_parameters_per_module(model)