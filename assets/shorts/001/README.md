# 初回ショート用素材

制作方式: 組み込みの `image_gen`。参照原画は[デザインver1](../../../docs/character-design/kyoko-design-ver1.png)。生成物は原本を上書きせず当ディレクトリへ保存した。衣装・人物は原画に従い、部屋は当作品の舞台案。[ADR 0015](../../../docs/adr/0015-full-body-short-production.md)参照。

| 素材 | 用途・画角 |
|---|---|
| standing.png | 正面全身、手を前に合わせて立つ |
| speaking.png | 同じ立ち姿の会話表情。レンダラーは顔の内側のみ合成 |
| crouch.png | 正面寄りの斜め全身、膝を曲げてしゃがむ |
| bow.png | 正面全身、腰から前へ倒してお辞儀する |
| room.png | 床・左の机・窓を含む縦型背景 |

人物4点は透過PNG。背景は不透明。画像はビルトイン生成であり、レンダリング時に画像生成APIは呼ばない。

## 生成プロンプト

### standing.png

Use case: identity-preserve. Production animation sprite, transparent RGBA PNG. Reference is character identity and drawing style only. Draw ONE full body Mochizuki Kyoko standing facing camera, hands nervously clasped before her, shy small smile, black long wavy hair flowing sideways, heavy eyelids gray eyes, blush and tiny sweat drop, navy business suit jacket white blouse navy ribbon navy knee length skirt black tights black low heels. Adult tall slim woman, preserve exactly the reference outfit and chibi illustration proportions, clean bold dark outlines, gentle cel shading. Entire body and shoes visible with generous transparent margins. No text, no panels, no backdrop, no cast shadow. This is standing pose for a short film. Save project asset.

### crouch.png

Use case identity-preserve. Reference is the same character standing. Create a SINGLE full-body sprite of exactly this adult woman in a low crouching pose, knees deeply bent together to the side modestly, weight on heels, hands resting nervously on knees, shoulders slightly hunched, looking up shyly toward camera. Front three-quarter view, entire hair silhouette, entire body and both shoes visible, transparent background. Preserve reference bold clean anime chibi line style, exact black long wavy hair, heavy eyelids gray eyes, blush tiny sweat, navy business suit white blouse navy ribbon knee skirt black tights black low heels. No change of outfit or identity. Natural bent joints, skirt covers thighs, no text no panels no floor no shadow. Pose dedicated to hiding behind a desk.

### bow.png

Use case identity-preserve. Create one full-body sprite of this EXACT SAME adult woman making a polite shy bow toward camera. Hinge torso forward at hips about 25 degrees, head inclined, eyes closed, hands together in front of skirt, knees softly bent. Entire hair outline, body and both shoes visible. Same black long wavy hair, navy suit jacket white blouse navy ribbon knee-length navy skirt black tights black low heels, same chibi proportions and clean bold anime outlines, subtle blush and sweat. Transparent background, no floor no shadows no text no panels, generous clear margin. This must be a real bowed drawing with articulated posture, not a rotated or squashed standing picture.

### speaking.png

Use case identity-preserve. Edit target is reference image. Preserve EXACT identical standing character pose, framing, proportions, hands, clothing, hair and transparency. Change ONLY mouth to tiny open speaking smile, eyes to warmer slightly smiling expression. Full body adult woman black long wavy hair navy business suit white blouse navy ribbon navy knee skirt black tights low heels, clean bold anime outlines. No backdrop no text no floor no shadows. Need registered speaking expression variant for animation.

### room.png

Use case: illustration-story. Background plate for a vertical 9:16 anime short, no people. Cozy quiet room at dusk, soft warm lamp illumination from upper left and lavender moonlight through window upper right. Hand drawn clean bold dark outlines and soft cel shading matching a cute chibi anime character, muted navy cream dusty lavender palette. Camera straight on, low enough to show ample wooden floor in lower half, believable perspective. IMPORTANT: a small wooden writing desk ONLY on LEFT THIRD of frame, desktop at exactly 58 percent image height, desk front panel extends down to 83 percent, its right edge at 36 percent image width. Rest of center and right side EMPTY visible open wooden floor for a full-body standing character, floor extends to bottom. Background wall, tasteful bookshelf upper left, simple curtain beside upper right window, small potted plant near right wall. No chair in center, no objects blocking center floor, no text, no watermark. Vertical portrait composition.

生成背景の机の位置は指示と異なるため、実画像を確認して台本の前景ポリゴンと人物の配置を実測に合わせた。画像指示の位置をそのまま描画座標には流用しない。
