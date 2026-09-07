El programa principal es detect_actions_json.py, donde el output es un json que se filtrará más adelante. 
ejemplo: python3 detect_actions_json.py ~/videos/6/robos_3_clips_clip61_000000_000020_6/user_15361/poses_full.npy
detect_action_filtered_json.py filtra las acciones y deja sólo las muñecas y sus posiciones relativas
Programas de validación:
> python3 validate_realtime.py ~/[path to npy] //visualizar el esqueleto y los mensajes de acción
> python3 validate_realtime_full.py ~/[path to npy] //visualizar el esqueleto y mp4 junto con los mensajes de acción, puede haber desincronía
> python3 validate_realtime_wrist.py ~/[path to npy] //visualizar el esqueleto y los mensajes de acción pero filtrado a posición de las muñecas
> python3 validate_realtime_wrist_full.py ~/[path to npy] //visualizar el esqueleto y mp4 y los mensajes de acción pero filtrado a posición de las muñecas 




ejemplo de salida del programa principal
output: ~/output/1/12Diciembre2025_12Diciembre2025_Cervezas_Cervezas_manana_002920_002950_1/user_136_poses_full_actions.json

source	"/home/ignacio/Escritorio/Company/Tecnica_Heuristica_HAR/videos/1/12Diciembre2025_12Diciembre2025_Cervezas_Cervezas_manana_002920_002950_1/user_136/poses_full.npy"
fps	12.38
fps_source	"/home/ignacio/Escritorio/Company/Tecnica_Heuristica_HAR/videos/1/12Diciembre2025_12Diciembre2025_Cervezas_Cervezas_manana_002920_002950_1/meta.json"
num_frames	370
duration_s	29.886914
filter	null
num_segments	44
segments	
0	
action	"ANY_WRIST_NEAR_WAIST"
start_frame	0
end_frame	369
start	"00:00:00.000"
end	"00:00:29.887"
start_s	0.0JS:0
end_s	29.886914
duration_s	29.886914
code	0
feature	"wrist_waist_L|wrist_waist_R"
reason	"al menos una muñeca cerca de la cintura"
enter_threshold	null
exit_threshold	null
1	
action	"ARMS_CROSSED"
start_frame	0
end_frame	329
start	"00:00:00.000"
end	"00:00:26.656"
start_s	0.0JS:0
end_s	26.655897
duration_s	26.655897
code	1
feature	"cross_L&cross_R"
reason	"HEURISTIC: both wrists past midline in 2D"
enter_threshold	null
exit_threshold	null
2	
action	"ARM_DOWN_LEFT"
start_frame	0
end_frame	369
start	"00:00:00.000"
end	"00:00:29.887"
start_s	0.0JS:0
end_s	29.886914
duration_s	29.886914
code	0
feature	"wrist_height_sh_L"
reason	"wrist below shoulder (image y)"
enter_threshold	-0.35
etc
