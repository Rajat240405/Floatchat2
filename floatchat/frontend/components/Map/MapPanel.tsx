"use client";

import { useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  ZoomControl,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import { motion } from "framer-motion";
import { Globe, Crosshair } from "lucide-react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapData } from "@/types";

interface MapPanelProps {
  mapData: MapData[];
  selectedFloat: string | null;
  onSelectFloat: (floatId: string | null) => void;
}

// Custom SVG marker icons
function createMarkerIcon(selected: boolean): L.DivIcon {
  const color = selected ? "#0ea5e9" : "#38bdf8";
  const size = selected ? 28 : 22;
  const pulse = selected
    ? `<circle cx="${size / 2}" cy="${size / 2}" r="${size / 2 + 4}" fill="none" stroke="${color}" stroke-width="1.5" opacity="0.4">
        <animate attributeName="r" from="${size / 2}" to="${size / 2 + 10}" dur="1.5s" repeatCount="indefinite"/>
        <animate attributeName="opacity" from="0.6" to="0" dur="1.5s" repeatCount="indefinite"/>
       </circle>`
    : "";

  return L.divIcon({
    className: "custom-marker",
    html: `
      <svg width="${size + 20}" height="${size + 20}" viewBox="0 0 ${size + 20} ${size + 20}" style="overflow:visible">
        ${pulse}
        <circle cx="${(size + 20) / 2}" cy="${(size + 20) / 2}" r="${size / 2}" fill="${color}" stroke="#0c4a6e" stroke-width="2.5"/>
        <circle cx="${(size + 20) / 2}" cy="${(size + 20) / 2}" r="${size / 4}" fill="white" opacity="0.9"/>
      </svg>
    `,
    iconSize: [size + 20, size + 20],
    iconAnchor: [(size + 20) / 2, (size + 20) / 2],
    popupAnchor: [0, -(size / 2 + 4)],
  });
}

// Auto-fit bounds to all markers
function BoundsFitter({ mapData }: { mapData: MapData[] }) {
  const map = useMap();

  useEffect(() => {
    if (mapData.length === 0) return;
    if (mapData.length === 1) {
      map.setView([mapData[0].latitude, mapData[0].longitude], 6, {
        animate: true,
        duration: 0.8,
      });
      return;
    }
    const bounds = L.latLngBounds(
      mapData.map((m) => [m.latitude, m.longitude])
    );
    map.fitBounds(bounds, { padding: [40, 40], animate: true, duration: 0.8 });
  }, [map, mapData]);

  return null;
}

export function MapPanel({ mapData, selectedFloat, onSelectFloat }: MapPanelProps) {
  const markerCount = mapData.length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="relative h-full bg-surface-900/50 border border-surface-800/60 rounded-xl overflow-hidden"
    >
      {/* Panel Header */}
      <div className="absolute top-0 left-0 right-0 z-[400] flex items-center gap-2 px-4 py-3 bg-surface-950/70 backdrop-blur-sm border-b border-surface-800/40">
        <Globe className="w-4 h-4 text-ocean-400" />
        <span className="text-sm font-medium text-surface-300">Global View</span>
        {markerCount > 0 && (
          <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-ocean-500/10 text-ocean-400 border border-ocean-500/20">
            {markerCount} float{markerCount !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Selected float indicator */}
      {selectedFloat && (
        <div className="absolute top-14 left-4 z-[400] flex items-center gap-2 px-3 py-1.5 rounded-lg bg-ocean-500/10 border border-ocean-500/20 backdrop-blur-sm">
          <Crosshair className="w-3.5 h-3.5 text-ocean-400" />
          <span className="text-xs font-medium text-ocean-300">
            Float {selectedFloat}
          </span>
          <button
            onClick={() => onSelectFloat(null)}
            className="ml-1 text-ocean-400 hover:text-ocean-200 text-xs"
          >
            ✕
          </button>
        </div>
      )}

      <MapContainer
        center={[20, 0]}
        zoom={2}
        zoomControl={false}
        scrollWheelZoom={true}
        className="h-full w-full"
        style={{ background: "#0f172a" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ZoomControl position="bottomright" />
        <BoundsFitter mapData={mapData} />

        {mapData.map((marker) => {
          const isSelected = selectedFloat === marker.float_id;
          return (
            <Marker
              key={marker.float_id}
              position={[marker.latitude, marker.longitude]}
              icon={createMarkerIcon(isSelected)}
              eventHandlers={{
                click: () => onSelectFloat(marker.float_id),
              }}
            >
              <Popup className="custom-popup">
                <div className="text-surface-100 min-w-[180px]">
                  <p className="font-semibold text-sm mb-1">
                    Float {marker.float_id}
                  </p>
                  <p className="text-xs text-surface-400 mb-1">
                    {marker.profile_date
                      ? new Date(marker.profile_date).toLocaleDateString()
                      : "Unknown date"}
                  </p>
                  <p className="text-xs text-surface-500 mb-1">
                    DAC: {marker.dac.toUpperCase()}
                  </p>
                  <p className="text-xs text-ocean-400">
                    {marker.variables.slice(0, 6).join(", ")}
                  </p>
                  {marker.variables.length > 6 && (
                    <p className="text-[10px] text-surface-600">
                      +{marker.variables.length - 6} more
                    </p>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </motion.div>
  );
}
