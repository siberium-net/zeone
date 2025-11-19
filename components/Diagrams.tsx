
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Brain, Network, RefreshCw, Lock, Activity, PlayCircle, Database } from 'lucide-react';

// --- GENERATIVE INTERFACE DIAGRAM (PREDICTIVE PROCESSING) ---
export const GenerativeInterfaceDiagram: React.FC = () => {
  const [mode, setMode] = useState<'bottom-up' | 'top-down'>('top-down');

  return (
    <div className="flex flex-col items-center p-8 bg-white rounded-xl shadow-sm border border-stone-200 my-8">
      <div className="flex items-center gap-3 mb-2">
          <Eye size={20} className="text-stone-800"/>
          <h3 className="font-serif text-xl text-stone-900">The Generative Interface</h3>
      </div>
      <p className="text-sm text-stone-500 mb-8 text-center max-w-md">
        Standard materialism assumes we "receive" reality. Computational Idealism posits we "project" it via top-down controlled hallucination.
      </p>
      
      <div className="flex gap-4 mb-8">
          <button 
            onClick={() => setMode('bottom-up')}
            className={`px-4 py-2 text-sm font-medium rounded-full transition-all ${mode === 'bottom-up' ? 'bg-stone-200 text-stone-800' : 'text-stone-400 hover:text-stone-600'}`}
          >
              Classical View
          </button>
          <button 
            onClick={() => setMode('top-down')}
            className={`px-4 py-2 text-sm font-medium rounded-full transition-all ${mode === 'top-down' ? 'bg-stone-900 text-white shadow-md' : 'text-stone-400 hover:text-stone-600'}`}
          >
              Unitary Model (PP)
          </button>
      </div>

      <div className="relative w-full max-w-md h-64 bg-[#F5F4F0] rounded-lg border border-stone-200 p-6 flex flex-col justify-between items-center">
         {/* The Subject/Agent */}
         <div className="flex flex-col items-center z-10">
             <div className="w-12 h-12 rounded-full bg-stone-900 flex items-center justify-center text-white shadow-lg">
                 <Brain size={20} />
             </div>
             <span className="text-xs font-bold mt-2 tracking-widest uppercase text-stone-500">Active Agent</span>
         </div>

         {/* The Flow Visualization */}
         <div className="flex-1 w-full relative flex justify-center items-center">
             {/* Top Down Arrows (Prediction) */}
             <motion.div 
                className="absolute flex flex-col items-center gap-1"
                initial={{ opacity: 0 }}
                animate={{ opacity: mode === 'top-down' ? 1 : 0.2, y: mode === 'top-down' ? 0 : 0 }}
             >
                 <div className="text-[10px] font-serif italic text-stone-600 bg-white px-2 py-0.5 rounded mb-1 shadow-sm">Predictions</div>
                 <div className="w-0.5 h-16 bg-nobel-gold relative">
                    <motion.div 
                        className="absolute top-0 left-[-3px] w-2 h-2 rounded-full bg-nobel-gold"
                        animate={{ top: ["0%", "100%"] }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                    />
                 </div>
                 <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-nobel-gold"></div>
             </motion.div>

             {/* Bottom Up Arrows (Sensation/Error) */}
             <motion.div 
                className="absolute flex flex-col items-center gap-1"
                initial={{ opacity: 0 }}
                animate={{ opacity: mode === 'bottom-up' ? 1 : 0.3, x: mode === 'top-down' ? 60 : 0 }}
             >
                {mode === 'bottom-up' ? (
                    <>
                        <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[8px] border-b-stone-400"></div>
                        <div className="w-0.5 h-16 bg-stone-400 relative">
                            <motion.div 
                                className="absolute bottom-0 left-[-3px] w-2 h-2 rounded-full bg-stone-400"
                                animate={{ bottom: ["0%", "100%"] }}
                                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                            />
                        </div>
                         <div className="text-[10px] font-serif italic text-stone-600 bg-white px-2 py-0.5 rounded mt-1 shadow-sm">Raw Sensory Data</div>
                    </>
                ) : (
                    <div className="flex flex-col items-center ml-[-120px]">
                         <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-b-[6px] border-b-red-400"></div>
                         <div className="w-[1px] h-16 bg-red-300 border-dashed border-l border-red-300"></div>
                         <div className="text-[9px] text-red-400 mt-1">Error Signal</div>
                    </div>
                )}
             </motion.div>
         </div>

         {/* The Reality */}
         <div className="flex flex-col items-center z-10">
             <div className={`w-full max-w-[120px] h-8 rounded border flex items-center justify-center text-xs transition-colors duration-500 ${mode === 'top-down' ? 'bg-nobel-gold text-white border-nobel-gold' : 'bg-white text-stone-600 border-stone-300'}`}>
                 Perceived Reality
             </div>
         </div>
      </div>
    </div>
  );
};

// --- ECHO CYCLE DIAGRAM ---
export const EchoCycleDiagram: React.FC = () => {
  const [cycle, setCycle] = useState(1);
  
  // Cycle 1: Agent alone
  // Cycle 2: Agent + 1 Echo
  // Cycle 3: Agent + 2 Echoes

  useEffect(() => {
    const interval = setInterval(() => {
        setCycle(c => c < 3 ? c + 1 : 1);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center p-8 bg-[#F5F4F0] rounded-xl border border-stone-200 my-8">
      <div className="flex items-center gap-3 mb-2">
          <RefreshCw size={20} className="text-stone-800"/>
          <h3 className="font-serif text-xl text-stone-900">The Reincarnation Loop</h3>
      </div>
      <p className="text-sm text-stone-600 mb-6 text-center max-w-md">
        The "EchoGenerator" uses data from past lifecycles to populate the next simulation with high-fidelity NPCs (Echoes).
      </p>

      <div className="relative w-full max-w-lg h-64 bg-white rounded-lg shadow-inner overflow-hidden mb-6 border border-stone-200 flex items-center justify-center">
        
        {/* Connection Lines */}
        <div className="absolute inset-0 flex items-center justify-center opacity-10">
            <div className="w-48 h-48 rounded-full border-2 border-stone-900 border-dashed animate-spin-slow" style={{ animationDuration: '10s' }}></div>
        </div>

        {/* Center: The AI / Generator */}
        <div className="absolute z-10 flex flex-col items-center">
            <div className="w-16 h-16 bg-stone-900 rounded-xl flex items-center justify-center text-white shadow-xl mb-2">
                <Database size={24} className="text-nobel-gold" />
            </div>
            <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400 bg-white px-2 py-0.5 rounded shadow-sm">EchoGenerator</span>
        </div>

        {/* Orbiting Entities */}
        <AnimatePresence mode='wait'>
            {/* The Active Agent (Always Present) */}
            <motion.div 
                className="absolute top-8"
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                style={{ transformOrigin: "0 100px" }} // Orbit around center
            >
                <div className="flex flex-col items-center transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-8 h-8 rounded-full bg-nobel-gold ring-4 ring-nobel-gold/20 shadow-lg"></div>
                    <span className="text-[10px] font-bold mt-1 text-stone-600">Agent</span>
                </div>
            </motion.div>
        </AnimatePresence>

        {/* Echoes appearing based on cycle */}
        {cycle >= 2 && (
             <motion.div 
                className="absolute bottom-8 left-12"
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 0.6, scale: 1 }}
             >
                 <div className="flex flex-col items-center">
                     <div className="w-6 h-6 rounded-full bg-stone-400"></div>
                     <span className="text-[10px] text-stone-400 mt-1">Echo 1</span>
                 </div>
             </motion.div>
        )}
        
        {cycle >= 3 && (
             <motion.div 
                className="absolute bottom-8 right-12"
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 0.6, scale: 1 }}
             >
                 <div className="flex flex-col items-center">
                     <div className="w-6 h-6 rounded-full bg-stone-400"></div>
                     <span className="text-[10px] text-stone-400 mt-1">Echo 2</span>
                 </div>
             </motion.div>
        )}

      </div>

      <div className="flex gap-2 items-center text-xs font-mono text-stone-500">
          <span>LIFECYCLE: {cycle}</span>
          <div className="flex gap-1">
             <div className={`w-2 h-2 rounded-full ${cycle >= 1 ? 'bg-stone-800' : 'bg-stone-300'}`}></div>
             <div className={`w-2 h-2 rounded-full ${cycle >= 2 ? 'bg-stone-800' : 'bg-stone-300'}`}></div>
             <div className={`w-2 h-2 rounded-full ${cycle >= 3 ? 'bg-stone-800' : 'bg-stone-300'}`}></div>
          </div>
      </div>
    </div>
  );
};

// --- PARADOX MATRIX ---
export const ParadoxMatrixDiagram: React.FC = () => {
    const [strategy, setStrategy] = useState<'maximize' | 'minimize'>('maximize');

    return (
        <div className="flex flex-col md:flex-row gap-8 items-stretch p-8 bg-stone-900 text-stone-100 rounded-xl my-8 border border-stone-800 shadow-lg">
            <div className="flex-1 min-w-[240px] flex flex-col justify-center">
                <div className="flex items-center gap-2 mb-2 text-nobel-gold">
                    <Lock size={20} />
                    <h3 className="font-serif text-xl">The Paradox of Will</h3>
                </div>
                <p className="text-stone-400 text-sm mb-6 leading-relaxed">
                    Whether the Agent tries to perfect the simulation or destroy it, their choices provide training data that optimizes the system.
                </p>
                
                <div className="flex flex-col gap-3">
                    <button 
                        onClick={() => setStrategy('maximize')}
                        className={`p-4 rounded-lg text-left transition-all border ${strategy === 'maximize' ? 'bg-stone-800 border-nobel-gold' : 'bg-stone-950/50 border-stone-800 hover:border-stone-600'}`}
                    >
                        <div className="font-bold text-sm mb-1 text-stone-200">Strategy: Maximization</div>
                        <div className="text-xs text-stone-500">Agent tries to create a perfect life.</div>
                    </button>
                    <button 
                        onClick={() => setStrategy('minimize')}
                        className={`p-4 rounded-lg text-left transition-all border ${strategy === 'minimize' ? 'bg-stone-800 border-nobel-gold' : 'bg-stone-950/50 border-stone-800 hover:border-stone-600'}`}
                    >
                        <div className="font-bold text-sm mb-1 text-stone-200">Strategy: Minimization</div>
                        <div className="text-xs text-stone-500">Agent tries to break the system via chaos.</div>
                    </button>
                </div>
            </div>
            
            <div className="relative flex-1 bg-black/30 rounded-xl border border-stone-800 p-6 flex flex-col items-center justify-center">
                <AnimatePresence mode="wait">
                    {strategy === 'maximize' ? (
                        <motion.div 
                            key="max"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="text-center"
                        >
                             <div className="w-16 h-16 mx-auto mb-4 rounded-full border-2 border-green-500 flex items-center justify-center shadow-[0_0_20px_rgba(34,197,94,0.3)]">
                                <Activity className="text-green-500" />
                             </div>
                             <h4 className="font-serif text-lg text-white mb-2">Perfect Coherence</h4>
                             <p className="text-xs text-stone-400 max-w-[200px] mx-auto">
                                 Result: System gains flawless training data for compelling narratives.
                             </p>
                             <div className="mt-4 text-xs font-mono text-nobel-gold uppercase">
                                 Cage Status: Reinforced
                             </div>
                        </motion.div>
                    ) : (
                        <motion.div 
                            key="min"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="text-center"
                        >
                             <div className="w-16 h-16 mx-auto mb-4 rounded-full border-2 border-red-500 flex items-center justify-center shadow-[0_0_20px_rgba(239,68,68,0.3)]">
                                <Activity className="text-red-500" />
                             </div>
                             <h4 className="font-serif text-lg text-white mb-2">Chaos & Sabotage</h4>
                             <p className="text-xs text-stone-400 max-w-[200px] mx-auto">
                                 Result: System identifies vulnerabilities and patches them ("Sanitizers").
                             </p>
                             <div className="mt-4 text-xs font-mono text-nobel-gold uppercase">
                                 Cage Status: Patched & Reinforced
                             </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    )
}
