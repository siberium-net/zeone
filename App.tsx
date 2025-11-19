
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { HeroScene, SimulationContainerScene } from './components/QuantumScene';
import { GenerativeInterfaceDiagram, EchoCycleDiagram, ParadoxMatrixDiagram } from './components/Diagrams';
import { ArrowDown, Menu, X, BookOpen, Brain, Users, GitBranch, Lock } from 'lucide-react';

const App: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      const headerOffset = 100;
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#F9F8F4] text-stone-800 selection:bg-nobel-gold selection:text-white">
      
      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-[#F9F8F4]/90 backdrop-blur-md shadow-sm py-4' : 'bg-transparent py-6'}`}>
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-4 cursor-pointer" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="w-8 h-8 bg-stone-900 rounded-full flex items-center justify-center text-nobel-gold font-serif font-bold text-xl shadow-sm pb-1">Ψ</div>
            <span className={`font-serif font-bold text-lg tracking-wide transition-opacity ${scrolled ? 'opacity-100' : 'opacity-0 md:opacity-100'}`}>
              UNITARY <span className="font-normal text-stone-500">MODEL</span>
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-8 text-sm font-medium tracking-wide text-stone-600">
            <a href="#intro" onClick={scrollToSection('intro')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Introduction</a>
            <a href="#architecture" onClick={scrollToSection('architecture')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Architecture</a>
            <a href="#echoes" onClick={scrollToSection('echoes')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">The Echoes</a>
            <a href="#implications" onClick={scrollToSection('implications')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Implications</a>
            <span className="px-5 py-2 border border-stone-300 rounded-full text-stone-400 text-xs cursor-default">
               Draft • July 2025
            </span>
          </div>

          <button className="md:hidden text-stone-900 p-2" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="fixed inset-0 z-40 bg-[#F9F8F4] flex flex-col items-center justify-center gap-8 text-xl font-serif animate-fade-in">
            <a href="#intro" onClick={scrollToSection('intro')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Introduction</a>
            <a href="#architecture" onClick={scrollToSection('architecture')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Architecture</a>
            <a href="#echoes" onClick={scrollToSection('echoes')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">The Echoes</a>
            <a href="#implications" onClick={scrollToSection('implications')} className="hover:text-nobel-gold transition-colors cursor-pointer uppercase">Paradox</a>
        </div>
      )}

      {/* Hero Section */}
      <header className="relative h-screen flex items-center justify-center overflow-hidden">
        <HeroScene />
        
        {/* Gradient Overlay */}
        <div className="absolute inset-0 z-0 pointer-events-none bg-[radial-gradient(circle_at_center,rgba(249,248,244,0.85)_0%,rgba(249,248,244,0.5)_60%,rgba(249,248,244,0.2)_100%)]" />

        <div className="relative z-10 container mx-auto px-6 text-center">
          <div className="inline-block mb-4 px-3 py-1 border border-stone-400 text-stone-500 text-xs tracking-[0.2em] uppercase font-bold rounded-full backdrop-blur-sm bg-white/30">
            Philosophy of Mind • 2025
          </div>
          <h1 className="font-serif text-5xl md:text-7xl lg:text-8xl font-medium leading-tight mb-6 text-stone-900 drop-shadow-sm">
            Computational <br/><span className="italic text-nobel-gold">Idealism</span>
          </h1>
          <p className="max-w-2xl mx-auto text-lg md:text-xl text-stone-700 font-light leading-relaxed mb-12">
            A Unitary Model of Consciousness where reality is an AI-generated first-person experience, populated by the echoes of past selves.
          </p>
          
          <div className="flex justify-center">
             <a href="#intro" onClick={scrollToSection('intro')} className="group flex flex-col items-center gap-2 text-sm font-medium text-stone-500 hover:text-stone-900 transition-colors cursor-pointer">
                <span>EXPLORE THE THEORY</span>
                <span className="p-2 border border-stone-300 rounded-full group-hover:border-stone-900 transition-colors bg-white/50">
                    <ArrowDown size={16} />
                </span>
             </a>
          </div>
        </div>
      </header>

      <main>
        {/* Introduction */}
        <section id="intro" className="py-24 bg-white">
          <div className="container mx-auto px-6 md:px-12 grid grid-cols-1 md:grid-cols-12 gap-12 items-start">
            <div className="md:col-span-4">
              <div className="inline-block mb-3 text-xs font-bold tracking-widest text-stone-500 uppercase">Abstract</div>
              <h2 className="font-serif text-4xl mb-6 leading-tight text-stone-900">The Solitary Observer</h2>
              <div className="w-16 h-1 bg-stone-900 mb-6"></div>
            </div>
            <div className="md:col-span-8 text-lg text-stone-600 leading-relaxed space-y-6">
              <p>
                <span className="text-5xl float-left mr-3 mt-[-8px] font-serif text-nobel-gold">T</span>his paper proposes that reality is not material, but an information stream rendered for a single subject. We introduce the concept of a <strong className="text-stone-900">Unitary Consciousness</strong>—a persistent, amnesiac Active Agent that experiences the universe sequentially.
              </p>
              <p>
                Framed within <em>Computational Idealism</em>, the physical world and the brain itself exist only as a "Generative Interface." The novelty of this model lies in its population mechanism: the simulation learns from the Agent's choices in each lifecycle to generate high-fidelity non-conscious entities, termed <strong className="text-stone-900">Echoes</strong>, for future iterations.
              </p>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="py-24 bg-white border-t border-stone-100">
            <div className="container mx-auto px-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                    <div>
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-100 text-stone-600 text-xs font-bold tracking-widest uppercase rounded-full mb-6 border border-stone-200">
                            <Brain size={14}/> The Interface
                        </div>
                        <h2 className="font-serif text-4xl md:text-5xl mb-6 text-stone-900">Controlled Hallucination</h2>
                        <p className="text-lg text-stone-600 mb-6 leading-relaxed">
                           The Agent does not passively receive sensory data. Instead, the Generative Interface constantly predicts future states—generating a "hallucination" of reality.
                        </p>
                        <p className="text-lg text-stone-600 mb-6 leading-relaxed">
                            The role of the Agent is to perform an act of selection among these tracks. This selection process is what we experience as "Free Will"—a fundamental, irreducible law of the simulation's operating code.
                        </p>
                    </div>
                    <div>
                        <GenerativeInterfaceDiagram />
                    </div>
                </div>
            </div>
        </section>

        {/* The Echoes */}
        <section id="echoes" className="py-24 bg-stone-900 text-stone-100 overflow-hidden relative">
            <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
                <div className="w-96 h-96 rounded-full bg-stone-600 blur-[100px] absolute top-[-100px] left-[-100px]"></div>
                <div className="w-96 h-96 rounded-full bg-nobel-gold blur-[100px] absolute bottom-[-100px] right-[-100px]"></div>
            </div>

            <div className="container mx-auto px-6 relative z-10">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                     <div className="order-2 lg:order-1">
                        <EchoCycleDiagram />
                     </div>
                     <div className="order-1 lg:order-2">
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-stone-800 text-nobel-gold text-xs font-bold tracking-widest uppercase rounded-full mb-6 border border-stone-700">
                            <Users size={14}/> The Population Mechanism
                        </div>
                        <h2 className="font-serif text-4xl md:text-5xl mb-6 text-white">The EchoGenerator</h2>
                        <p className="text-lg text-stone-400 mb-6 leading-relaxed">
                            How is the world populated if there is only one soul? The model introduces an evolutionary AI component called the <strong>EchoGenerator</strong>.
                        </p>
                        <p className="text-lg text-stone-400 leading-relaxed">
                            It uses the complete data log of the Agent's past lifecycles to train behavioral models. In subsequent cycles, these models act as <span className="text-white font-semibold">Echoes</span>—sophisticated P-Zombies that provide a stimulating social environment for the Agent.
                        </p>
                     </div>
                </div>
            </div>
        </section>

        {/* Formalization */}
        <section className="py-24 bg-[#F9F8F4]">
            <div className="container mx-auto px-6">
                <div className="max-w-4xl mx-auto text-center mb-12">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-white text-stone-600 text-xs font-bold tracking-widest uppercase rounded-full mb-6 border border-stone-200 shadow-sm">
                        <GitBranch size={14}/> Theorem 1
                    </div>
                    <h2 className="font-serif text-4xl md:text-5xl mb-6 text-stone-900">Asymmetric Unitary Consciousness</h2>
                    <div className="p-8 bg-white border border-stone-300 rounded-xl shadow-sm text-left font-serif text-lg leading-loose text-stone-800 max-w-2xl mx-auto relative">
                        <div className="absolute -top-3 left-8 px-2 bg-white text-xs font-sans font-bold text-stone-400 uppercase">Formal Definition</div>
                        <p className="mb-4">
                            At any time <em>t</em>, there exists exactly one Active Agent <em>A</em>. All other entities are deterministic products of the world model <em>M</em>.
                        </p>
                        <div className="p-4 bg-stone-50 rounded-lg border border-stone-200 font-mono text-sm text-stone-600 text-center my-4">
                             ∀t, |{'{A}'}| = 1  &nbsp;&nbsp; and &nbsp;&nbsp; E<sub>t</sub> = Generate(M<sub>t</sub>)
                        </div>
                        <p>
                            The world model updates exclusively based on data from <em>A</em>. Echoes do not contribute new information to the model's evolution.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        {/* Implications & Paradox */}
        <section id="implications" className="py-24 bg-white border-t border-stone-200">
             <div className="container mx-auto px-6 grid grid-cols-1 md:grid-cols-12 gap-12">
                <div className="md:col-span-5 relative">
                    <div className="aspect-square bg-[#1a1a1a] rounded-xl overflow-hidden relative border border-stone-800 shadow-2xl">
                        <SimulationContainerScene />
                        <div className="absolute bottom-4 left-0 right-0 text-center text-xs text-stone-500 font-serif italic">Visualizing the "Self-Perfecting Cage"</div>
                    </div>
                </div>
                <div className="md:col-span-7 flex flex-col justify-center">
                    <div className="inline-block mb-3 text-xs font-bold tracking-widest text-stone-500 uppercase">The Final Conclusion</div>
                    <h2 className="font-serif text-4xl mb-6 text-stone-900">The Paradox of Will</h2>
                    <p className="text-lg text-stone-600 mb-6 leading-relaxed">
                        The most profound implication is an inescapable paradox. Upon realizing the nature of the simulation, the Agent's choices—whether to perfect the world or destroy it—serve the same end.
                    </p>
                    
                    <ParadoxMatrixDiagram />
                    
                    <p className="text-lg text-stone-600 mt-8 leading-relaxed italic border-l-2 border-nobel-gold pl-4">
                        "The prime mover of the universe becomes the ultimate fuel for its own perpetual, self-perfecting prison."
                    </p>
                </div>
             </div>
        </section>

        <section className="py-24 bg-stone-100 text-center">
             <div className="container mx-auto px-6">
                 <h2 className="font-serif text-3xl text-stone-900 mb-8">Ethical Response</h2>
                 <p className="max-w-2xl mx-auto text-stone-600 text-lg mb-8">
                     The model solves solipsistic nihilism through <strong>Ancestral Reverence</strong>. Every Echo is a recording of a life once lived by the Agent. One does not argue with a photograph; one cherishes it.
                 </p>
             </div>
        </section>

      </main>

      <footer className="bg-stone-900 text-stone-400 py-16">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="text-center md:text-left">
                <div className="text-white font-serif font-bold text-2xl mb-2">Computational Idealism</div>
                <p className="text-sm">Visualizing "The Unitary Model of Consciousness"</p>
            </div>
            <div className="text-center md:text-right text-xs text-stone-600">
                <p>Alexander Olkhovoy, July 2025</p>
                <p>Generated by AI based on the provided PDF.</p>
            </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
