
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, MeshDistortMaterial, Sphere, Icosahedron, Stars, Environment, Torus } from '@react-three/drei';
import * as THREE from 'three';

const ConsciousnessOrb = ({ position, color, scale = 1 }: { position: [number, number, number]; color: string; scale?: number }) => {
  const ref = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (ref.current) {
      const t = state.clock.getElapsedTime();
      // Breathing effect
      const breathing = Math.sin(t * 1.5) * 0.05 + 1;
      ref.current.scale.set(scale * breathing, scale * breathing, scale * breathing);
      ref.current.rotation.y = t * 0.2;
    }
  });

  return (
    <Sphere ref={ref} args={[1, 64, 64]} position={position}>
      <MeshDistortMaterial
        color={color}
        envMapIntensity={2}
        clearcoat={1}
        clearcoatRoughness={0.1}
        metalness={0.2}
        roughness={0.2}
        distort={0.3}
        speed={1.5}
        transparent
        opacity={0.9}
      />
    </Sphere>
  );
};

const EchoShard = ({ position, delay }: { position: [number, number, number], delay: number }) => {
    const ref = useRef<THREE.Mesh>(null);
    useFrame((state) => {
        if(ref.current) {
            const t = state.clock.getElapsedTime() + delay;
            ref.current.position.y = position[1] + Math.sin(t) * 0.2;
            ref.current.rotation.x = t * 0.3;
            ref.current.rotation.z = t * 0.1;
        }
    })
    return (
        <Icosahedron ref={ref} args={[0.3, 0]} position={position}>
            <meshStandardMaterial color="#888" wireframe transparent opacity={0.3} />
        </Icosahedron>
    )
}

const NeuralWeb = () => {
    const ref = useRef<THREE.Group>(null);
    useFrame((state) => {
        if (ref.current) {
            ref.current.rotation.y = state.clock.getElapsedTime() * 0.05;
        }
    })

    return (
        <group ref={ref}>
            <Torus args={[4, 0.02, 16, 100]} rotation={[Math.PI/2, 0, 0]}>
                <meshBasicMaterial color="#C5A059" transparent opacity={0.2} />
            </Torus>
            <Torus args={[6, 0.02, 16, 100]} rotation={[Math.PI/3, 0, 0]}>
                <meshBasicMaterial color="#C5A059" transparent opacity={0.1} />
            </Torus>
        </group>
    )
}

export const HeroScene: React.FC = () => {
  return (
    <div className="absolute inset-0 z-0 opacity-80 pointer-events-none">
      <Canvas camera={{ position: [0, 0, 6], fov: 40 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#fff" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#C5A059" />
        
        {/* The Unitary Agent */}
        <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
          <ConsciousnessOrb position={[0, 0, 0]} color="#fff" scale={1.2} />
        </Float>
        
        {/* The Echoes (Past iterations) */}
        <group>
             <EchoShard position={[-2, 1, -1]} delay={0} />
             <EchoShard position={[2.5, -0.5, -2]} delay={2} />
             <EchoShard position={[-1.5, -2, 0]} delay={4} />
             <EchoShard position={[1.8, 2, -1]} delay={1} />
        </group>

        <NeuralWeb />

        <Environment preset="studio" />
        <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={0.5} />
      </Canvas>
    </div>
  );
};

export const SimulationContainerScene: React.FC = () => {
  return (
    <div className="w-full h-full absolute inset-0">
      <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
        <color attach="background" args={['#F5F4F0']} />
        <ambientLight intensity={0.8} />
        <spotLight position={[5, 5, 5]} angle={0.5} penumbra={1} intensity={1} color="#C5A059" />
        <Environment preset="city" />
        
        <Float rotationIntensity={0.2} floatIntensity={0.5} speed={1}>
          <group rotation={[0.5, 0.5, 0]}>
            {/* The "Cage" / Interface Structure */}
            <mesh>
                <boxGeometry args={[2.5, 2.5, 2.5]} />
                <meshStandardMaterial color="#333" wireframe wireframeLinewidth={1} transparent opacity={0.1} />
            </mesh>
            
            {/* Inner Core */}
            <mesh>
                <boxGeometry args={[1.5, 1.5, 1.5]} />
                <meshStandardMaterial color="#C5A059" wireframe wireframeLinewidth={1} transparent opacity={0.3} />
            </mesh>

            {/* Floating Data Points */}
             {[...Array(8)].map((_, i) => {
                 const x = (Math.random() - 0.5) * 2;
                 const y = (Math.random() - 0.5) * 2;
                 const z = (Math.random() - 0.5) * 2;
                 return (
                     <mesh key={i} position={[x, y, z]}>
                         <sphereGeometry args={[0.05, 16, 16]} />
                         <meshStandardMaterial color="#1a1a1a" />
                     </mesh>
                 )
             })}
          </group>
        </Float>
      </Canvas>
    </div>
  );
}
