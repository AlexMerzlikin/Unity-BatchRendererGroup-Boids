using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace BatchRendererGroupExample
{
    [BurstCompile]
    public unsafe struct CopyMatricesJob : IJobParallelFor
    {
        public int Size;

        [ReadOnly] [NativeDisableUnsafePtrRestriction]
        public float4x4* Source;

        [WriteOnly] [NativeDisableParallelForRestriction]
        public NativeArray<Vector4> DataBuffer;

        private const int PositionOffset = 4;

        public void Execute(int index)
        {
            DataBuffer[PositionOffset + index * 3 + 0] = new Vector4(Source[index].c0.x, Source[index].c0.y, Source[index].c0.z, Source[index].c1.x);
            DataBuffer[PositionOffset + index * 3 + 1] = new Vector4(Source[index].c1.y, Source[index].c1.z, Source[index].c2.x, Source[index].c2.y);
            DataBuffer[PositionOffset + index * 3 + 2] = new Vector4(Source[index].c2.z, Source[index].c3.x, Source[index].c3.y, Source[index].c3.z);

            var offset = Size * 3;
            var inverse = Matrix4x4.Inverse(Source[index]);
            DataBuffer[PositionOffset + index * 3 + 0 + offset] = new Vector4(inverse.m00, inverse.m10, inverse.m20, inverse.m01);
            DataBuffer[PositionOffset + index * 3 + 1 + offset] = new Vector4(inverse.m11, inverse.m21, inverse.m02, inverse.m12);
            DataBuffer[PositionOffset + index * 3 + 2 + offset] = new Vector4(inverse.m22, inverse.m03, inverse.m13, inverse.m23);
        }
    }
}
