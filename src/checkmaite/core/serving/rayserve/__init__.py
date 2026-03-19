"""Ray Serve integration for remote model inference."""

from checkmaite.core.serving.rayserve._clients import (
    RayImageClassificationClient,
    RayObjectDetectionClient,
)
from checkmaite.core.serving.rayserve._deployments import (
    RayImageClassificationDeployment,
    RayObjectDetectionDeployment,
)

__all__ = [
    "RayImageClassificationClient",
    "RayImageClassificationDeployment",
    "RayObjectDetectionClient",
    "RayObjectDetectionDeployment",
    "print_serve_status",
]


def print_serve_status() -> None:
    """Print the status of all Ray Serve applications and their deployments."""
    from ray import serve

    try:
        status = serve.status()
    except RuntimeError as e:
        print(
            f"\nNote: Could not query cluster status ({type(e).__name__}). "
            "Make sure Ray is running with `ray start --head`"
        )
        return

    print(f"\n{'=' * 70}")
    print("Ray Serve Cluster Status")
    print(f"{'=' * 70}\n")

    if not status.applications:
        print("No applications are currently deployed.")
    else:
        for app_name, app_info in status.applications.items():
            print(f"Application: {app_name}")
            print(f"  Status: {app_info.status}")

            for deployment_name, dep_info in app_info.deployments.items():
                replica_count = len(dep_info.replica_states)
                healthy_replicas = sum(1 for s in dep_info.replica_states.values() if "healthy" in str(s).lower())

                print(f"\n  Deployment: {deployment_name}")
                print(f"    Replicas: {replica_count} total, {healthy_replicas} healthy")

            print()

    print("=" * 70)
