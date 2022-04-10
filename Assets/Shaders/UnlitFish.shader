Shader "CustomURP/Unlit Fish"
{
    Properties
    {
        _BaseMap ("Base Texture", 2D) = "white" {}
        _BaseColor ("Base Colour", Color) = (0, 0.66, 0.73, 1)
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline"="UniversalPipeline" "Queue"="Geometry"
        }

        Pass
        {
            Name "Forward"
            Tags
            {
                "LightMode"="UniversalForward"
            }

            Cull Back

            HLSLPROGRAM
            #pragma exclude_renderers gles gles3 glcore
            #pragma target 4.5
            #pragma vertex UnlitPassVertex
            #pragma fragment UnlitPassFragment
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer
            #pragma multi_compile _ DOTS_INSTANCING_ON
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            float4 _BaseColor;
            CBUFFER_END

            #ifdef UNITY_DOTS_INSTANCING_ENABLED
                UNITY_DOTS_INSTANCING_START(MaterialPropertyMetadata)
                    UNITY_DOTS_INSTANCED_PROP(float4, _BaseColor)
                UNITY_DOTS_INSTANCING_END(MaterialPropertyMetadata)
                #define _BaseColor UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(float4, _BaseColor)
            #endif
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            Varyings UnlitPassVertex(Attributes input)
            {
                Varyings output;

                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                half _EffectRadius = 0;
                half _WaveSpeed = 3.5;
                half _WaveHeight = 0.25;
                half _WaveDensity = 0.69;
                half _Yoffset = 0.89;
                int _Threshold = 0.06;

                half sinUse = sin(-_Time.y * _WaveSpeed + input.positionOS.z * _WaveDensity);
                half yValue = input.positionOS.y - _Yoffset;
                half yDirScaling = clamp(pow(yValue * _EffectRadius, _Threshold), 0.0, 1.0);
                input.positionOS.x = input.positionOS.x + sinUse * _WaveHeight * yDirScaling;

                const VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.color = input.color;
                return output;
            }

            half4 UnlitPassFragment(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                const half4 baseMap = half4(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv));
                return baseMap * _BaseColor * input.color;
            }
            ENDHLSL
        }
    }
}