"use client"

import { useEffect, useRef } from "react"

interface Particle {
  x: number
  y: number
  size: number
  color: string
  duration: number
  delay: number
}

const COLORS = ["hsl(var(--accent))", "hsl(326 100% 62%)", "hsl(43 100% 52%)"]

const PARTICLES: Particle[] = Array.from({ length: 24 }, (_, i) => ({
  x: 55 + ((i * 37) % 40),
  y: 5 + ((i * 53) % 80),
  size: 2 + (i % 4),
  color: COLORS[i % 3],
  duration: 6 + (i % 6),
  delay: -(i * 0.6),
}))

export default function HeroAmbient() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleMouse = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / rect.width - 0.5) * 2
      const y = ((e.clientY - rect.top) / rect.height - 0.5) * 2
      const rotateX = -y * 3
      const rotateY = x * 3
      container.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`
    }

    const handleLeave = () => {
      container.style.transform = "rotateX(0deg) rotateY(0deg)"
    }

    const parent = container.parentElement
    if (parent) {
      parent.addEventListener("mousemove", handleMouse)
      parent.addEventListener("mouseleave", handleLeave)
    }

    return () => {
      if (parent) {
        parent.removeEventListener("mousemove", handleMouse)
        parent.removeEventListener("mouseleave", handleLeave)
      }
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="pointer-events-none absolute inset-0 -z-5 overflow-hidden"
      style={{
        perspective: "800px",
        transformStyle: "preserve-3d",
        transition: "transform 0.15s ease-out",
      }}
    >
      {PARTICLES.map((p, i) => (
        <div
          key={i}
          className="absolute rounded-full"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            backgroundColor: p.color,
            boxShadow: p.size > 3 ? `0 0 ${p.size * 3}px ${p.color}` : undefined,
            opacity: 0,
            animation: `heroParticleFloat ${p.duration}s ease-in-out ${p.delay}s infinite, heroParticleFade ${p.duration * 0.4}s ${p.delay}s forwards`,
            willChange: "transform",
          }}
        />
      ))}

      {/* Central glowing orb */}
      <div
        className="absolute rounded-full"
        style={{
          left: "72%",
          top: "35%",
          width: 200,
          height: 200,
          background: "radial-gradient(circle at center, hsl(var(--accent)), transparent 70%)",
          opacity: 0.06,
          animation: "heroOrbPulse 6s ease-in-out infinite alternate",
        }}
      />

      {/* Secondary magenta orb */}
      <div
        className="absolute rounded-full"
        style={{
          left: "82%",
          top: "55%",
          width: 140,
          height: 140,
          background: "radial-gradient(circle at center, hsl(326 100% 62%), transparent 70%)",
          opacity: 0.04,
          animation: "heroOrbPulse 8s ease-in-out 1s infinite alternate",
        }}
      />
    </div>
  )
}
