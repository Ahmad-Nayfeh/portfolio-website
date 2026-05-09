"use client"

/**
 * FadeIn — lightweight IntersectionObserver scroll-reveal wrapper.
 *
 * Uses CSS transitions (not Framer Motion) for zero bundle overhead.
 * Fades + lifts the children into view when they enter the viewport.
 *
 * Usage:
 *   <FadeIn delay={120}>
 *     <p>Content that fades up on scroll.</p>
 *   </FadeIn>
 */

import { useEffect, useRef, useState } from "react"
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export interface FadeInProps {
  children: ReactNode
  className?: string
  /** Animation delay in milliseconds. Useful for staggered siblings. */
  delay?: number
  /** IntersectionObserver threshold. Default 0.1. */
  threshold?: number
  /**
   * How far the element lifts on entry.
   * Accepts any valid CSS translateY value. Default "10px".
   */
  distance?: string
  /**
   * When true (default) the element stays visible once it first appears.
   * Set to false for a re-animate-on-scroll effect.
   */
  once?: boolean
  /** Transition duration in ms. Default 500. */
  duration?: number
}

export default function FadeIn({
  children,
  className,
  delay = 0,
  threshold = 0.1,
  distance = "10px",
  once = true,
  duration = 500,
}: FadeInProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Bail out early when reduced-motion is requested — render immediately.
    const query = window.matchMedia("(prefers-reduced-motion: reduce)")
    if (query.matches) {
      setVisible(true)
      return
    }

    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          if (once) observer.unobserve(el)
        } else if (!once) {
          setVisible(false)
        }
      },
      { threshold },
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold, once])

  return (
    <div
      ref={ref}
      className={cn(className)}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : `translateY(${distance})`,
        transition: `opacity ${duration}ms cubic-bezier(0.22,1,0.36,1) ${delay}ms, transform ${duration}ms cubic-bezier(0.22,1,0.36,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  )
}
