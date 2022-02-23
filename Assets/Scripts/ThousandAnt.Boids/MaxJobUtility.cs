using UnityEngine;
using Unity.Jobs.LowLevel.Unsafe;

namespace ThousandAnt.Boids {

    public class MaxJobUtility : MonoBehaviour {

#pragma warning disable 649
        [SerializeField]
        private ushort jobsCount = 4;
#pragma warning restore 649

        private void Awake() {
            JobsUtility.JobWorkerCount = jobsCount;
        }
    }
}
