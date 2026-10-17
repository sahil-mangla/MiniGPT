import os

STORIES = [
    """Once upon a time, there was a little girl named Lily. Lily loved to play in her green garden. One sunny morning, Lily saw a tiny blue bird sitting on a high wooden branch. The little bird looked cold and lonely.
"Hello, little bird," Lily said softly. "Are you hungry?"
The bird chirped cheerfully. Lily ran into the kitchen and found a slice of soft bread. She broke the bread into small crumbs and scattered them gently across the grass.
The bird hopped down from the branch. It pecked at the delicious crumbs happily.
"You are a good friend, little bird," Lily giggled.
After eating, the bird flapped its colorful wings and flew up into the blue sky. Lily felt warm and proud inside because she helped a hungry friend.""",

    """Tim had a shiny red ball. It was his favorite toy in the whole world. Every afternoon, Tim took the red ball to the park with his playful dog, Max.
Max loved chasing the ball. Tim threw the ball as far as he could across the soft grass.
"Go get it, Max!" shouted Tim with a big smile.
Max ran as fast as his four paws could carry him. He caught the red ball with his mouth and wagged his fluffy tail. But suddenly, the ball bounced into a thick green bush.
Tim walked over and peeked into the bush. He saw the red ball resting next to a sleepy brown frog.
"Careful, Max," whispered Tim. "Do not scare the frog."
Tim gently reached his hand in and picked up the ball. The frog blinked slowly and hopped away into the pond. Tim and Max ran home together, happy after a fun day of playing.""",

    """Once there was a little boy named Sam. Sam loved shiny rocks. Wherever he walked, his eyes looked down at the ground to spot sparkling stones.
One afternoon, Sam walked along a quiet dirt path near a bubbling stream. He saw something bright reflecting the sunlight under a willow tree.
"Look at that!" Sam gasped.
He knelt down and picked up a smooth, purple stone. It was cool to the touch and perfectly round.
Just then, his little sister Mia came running over. She tripped on a small tree root and began to cry.
Sam ran to her side. "Do not cry, Mia. Look at this magical purple stone I found!"
Mia stopped crying and wiped her tears. She touched the smooth stone and smiled.
"Can I hold it, Sam?" asked Mia.
"You can keep it, Mia," Sam replied kindly. Giving the treasure to his sister made Sam even happier than finding it.""",

    """In a cozy yellow house on the hill lived a curious cat named Oliver. Oliver had soft white fur and bright green eyes. Oliver liked to explore every corner of the house.
One chilly evening, Oliver saw a closed wooden door at the end of the long hallway. The door was usually locked, but tonight it was open just a crack.
Oliver tiptoed forward on silent paws. He pushed the door open with his wet pink nose.
Inside the room was a giant bookshelf filled with old storybooks and a cozy armchair. On the chair sat a warm woolen blanket.
Oliver jumped up onto the chair. He curled into a little ball on top of the blanket. He began to purr loudly as the gentle rain tapped against the window glass. Oliver fell fast asleep, dreaming of chasing bright golden butterflies.""",

    """Lucy had a box of bright wax crayons. There was red, yellow, green, blue, and her favorite color, bright purple. Lucy loved to draw pictures for her grandmother.
Lucy sat down at the wooden table with a large sheet of white paper.
"I will draw a tall castle with a friendly green dragon," Lucy said to herself.
She drew the castle walls with grey stones. She drew tall orange flags on top of the towers. Next to the castle, she drew a smiling green dragon holding a bunch of yellow flowers.
When her grandmother walked into the room, she gasped with joy.
"Oh Lucy, this is the most wonderful drawing I have ever seen!" grandmother exclaimed.
Grandmother hung the colorful picture on the refrigerator door for everyone to see. Lucy felt very proud of her artwork.""",

    """Leo was a small turtle who lived near a wide blue river. While the other animals ran and jumped fast, Leo walked very, very slowly. Sometimes Leo wished he could run like the rabbits.
One hot summer day, a busy brown rabbit named Benny came running past.
"Hurry up, Leo! You are so slow!" Benny teased as he sped ahead.
Leo sighed and kept walking at his steady pace. Step, step, step.
Soon, dark clouds filled the sky and cold rain began to pour down heavily. Benny the rabbit had no shelter and got soaking wet. He shivered under a tiny leaf.
When Leo caught up, he pulled his head and legs safely inside his hard, dry shell.
"You can stand close under my shell, Benny," offered kind Leo.
Benny sheltered from the wind until the storm passed.
"Thank you, Leo," said Benny softly. "Being slow and having a strong shell is wonderful." From that day on, Benny never teased Leo again.""",

    """Tommy and his father decided to build a wooden birdhouse. Tommy put on his small blue work gloves and held the ruler while his father cut the clean pine boards.
"Hold the wood steady, Tommy," his father said gently.
Tommy held the board carefully with both hands. His father tapped the nails with a hammer: tap, tap, tap!
Together, they painted the birdhouse bright yellow with a cheerful red roof. They hung the finished house on an oak branch in the front yard.
A few days later, two small yellow finches arrived. They carried soft twigs and moss in their beaks into the little round entrance hole.
"Look, Dad!" Tommy whispered excitedly. "They made it their home!"
Tommy felt overjoyed that his hard work gave the birds a safe and warm nest.""",

    """Mia had a baby doll named Bella. Bella wore a pink polka-dot dress and had painted blue eyes. Mia took Bella everywhere she went.
One afternoon, Mia went shopping at the grocery store with her mother. Mia sat Bella inside the shopping cart.
When they returned home and unpacked the groceries, Mia noticed something terrible: Bella was gone!
Mia began to cry. "Mom, where is Bella? Did we leave her at the store?"
Mother held Mia's hand. "Do not worry, dear. Let us go back and check the store."
They drove back to the market. The kind store manager was standing by the front door holding the little doll.
"A nice boy found her in a shopping cart and brought her to me," the manager said with a smile.
Mia hugged Bella tightly against her chest. "Thank you so much!" she cheered. Mia promised to always take good care of her beloved toys.""",

    """Once there was a little bear cub named Barnaby. Barnaby was terrified of the dark. When the sun went down behind the mountain, Barnaby hid under his mossy bed.
"Mother," Barnaby cried, "the night is too scary!"
His mother sat beside him and stroked his fuzzy brown ears.
"Come outside with me for a moment, Barnaby," she whispered.
Barnaby crept out of the cave cautiously. He looked up at the night sky. Thousands of twinkling silver stars danced across the deep velvet blue, and a huge round moon glowed like a giant lantern.
Fireflies floated above the meadow grass like tiny floating lights.
"You see, Barnaby?" his mother said gently. "The dark is not empty. It has its own gentle lights to guide us."
Barnaby looked at the glowing fireflies and yawned. He was not afraid anymore, and he slept peacefully through the starry night.""",

    """Ella loved visiting her grandfather's farm during the harvest season. The farm had tall apple trees full of sweet, crunchy red fruit.
"Grandpa, can we bake an apple pie today?" Ella asked, jumping up and down.
"Only if you help me pick the biggest apples, Ella," Grandpa chuckled.
They took two woven wicker baskets and walked out into the sunny orchard. Ella reached up on her tiptoes and plucked the ripest red apples from the low branches.
In the warm kitchen, Grandpa showed Ella how to roll the soft flour dough flat with a wooden rolling pin. They sprinkled sweet brown cinnamon over the sliced apples and baked the pie in the oven.
The entire house smelled delicious. When the pie cooled down, they shared a big warm slice together with cold milk. It was the sweetest afternoon ever.""",

    """Jack had a red tricycle with a ringing silver bell. Jack loved to pedal up and down the paved driveway.
Ring, ring! went the bell as Jack sped past the flowerbeds.
One morning, Jack noticed a trail of tiny black ants marching across the concrete. They were carrying breadcrumbs on their backs to an anthill near the lawn.
Jack stopped his tricycle immediately.
"I must not squish the hard-working ants," Jack thought to himself.
He waited patiently until the entire line of ants crossed safely onto the grass. Then he pedaled forward with a smile. Jack knew that even the smallest creatures deserved kindness and respect.""",

    """Emma found an old cardboard box in the garage. To anyone else, it looked like trash, but to Emma, it was a magnificent spaceship!
She took her colored markers and drew dials, switches, and shiny stars on the cardboard flaps. She cut two round holes for windows.
"Prepare for blastoff in three, two, one!" Emma shouted.
Emma climbed inside the box and shook it gently as if roaring through outer space. She looked through the cardboard windows and imagined zooming past the moon and floating among glittering comets.
Her mother opened the door holding a tray of warm oatmeal cookies.
"Astronaut Emma, your space fuel has arrived," her mother smiled.
Emma giggled and munched on a cookie, eager for her next galactic adventure.""",

    """Ben had a little green garden patch behind his house. In the spring, he planted a tiny sunflower seed deep in the rich brown soil.
Every single day after school, Ben carried a small watering can to his garden.
"Drink water and grow tall, little seed," Ben spoke encouragingly.
Weeks passed. First, two tiny green leaves poked up from the dirt. Then a sturdy green stem shot toward the sky.
By midsummer, the sunflower stood taller than Ben himself! On top blossomed a gigantic golden flower that turned its cheerful face toward the warm sun.
Bright yellow bumblebees came to drink nectar from the sunflower's center. Ben smiled proudly, knowing that patience and love made the beautiful plant thrive.""",

    """Sophie was nervous about her first day at nursery school. She clung tightly to her father's coat in the classroom doorway.
"What if nobody wants to be my friend?" Sophie asked softly, looking down at her shoes.
Her father knelt down and kissed her forehead. "Just smile and share, Sophie. You will do wonderfully."
Sophie took a deep breath and walked to the wooden block table. Another girl with curly brown hair was trying to build a tall tower, but the blocks kept tumbling down.
"Would you like some help?" Sophie asked timidly.
The girl looked up and beamed. "Yes please! My name is Chloe."
Together, Sophie and Chloe built a magnificent castle with four sturdy towers. Sophie forgot all about being nervous, and she realized that making a new friend was as simple as sharing a smile.""",

    """A tiny white mouse named Pip lived beneath the floorboards of a busy bakery. Every morning, the sweet scent of fresh warm bread and cinnamon rolls filled the air.
Pip was very careful never to be seen by the baker. But today, the baker accidentally dropped a golden raisin on the clean tile floor.
Pip peeked out from his mousehole. His whiskers twitched with excitement.
"That raisin looks so sweet and plump," Pip thought.
He scampered across the floor on silent tiny paws, grabbed the raisin in his teeth, and dashed back into his safe little home just in time.
Pip nibbled on the raisin slowly, savoring every sweet bite. It was the greatest feast of his whole week.""",

    """Noah had a blue sailboat that he carved out of a dry piece of cedar wood with his older brother.
On Saturday morning, they walked to the clear neighborhood duck pond. Noah placed the small wooden boat into the water.
A gentle breeze caught the little cloth sail. The boat glided gracefully across the rippling water, leaving a small wake behind it.
Two curious white ducks swam over to inspect the tiny vessel. They quacked cheerfully and escorted the boat across the pond.
Noah ran along the grassy edge, clapping his hands with excitement. The little boat sailed all the way to the other side without tipping over once. Noah picked it up, dried the wood with his shirt, and felt like a true sea captain.""",

    """Ruby was a fluffy golden retriever puppy who loved to dig in the backyard dirt. Her owner, Lucas, always told her: "Ruby, no digging in the flowerbeds!"
One sunny morning, Ruby smelled something interesting near the big oak tree. She could not resist. Her front paws dug into the damp soil: dig, dig, dig!
Suddenly, her paw struck something hard. It was not a rock. It was Lucas's favorite blue toy car that he lost last autumn!
Ruby picked up the toy car in her mouth and trotted proudly to the porch where Lucas was sitting. She dropped the car right into his lap.
"Ruby, you found my car!" Lucas yelled with joy, hugging the dusty puppy.
Lucas washed the toy car, brushed the dirt off Ruby's paws, and gave her a tasty bone for being such a wonderful detective.""",

    """On a cold winter morning, Liam looked out his bedroom window. The world was completely covered in a thick blanket of soft white snow!
Liam put on his thick wool socks, his warm blue jacket, his red mittens, and his cozy knit hat. He ran outside into the fresh crisp air.
His boots went crunch, crunch, crunch in the deep snow.
Liam rolled a small snowball along the ground. As he rolled it, the ball grew bigger and bigger until it was heavy and round. He made another ball for the middle and a third ball for the head.
Liam used two black stones for eyes, a crunchy orange carrot for the nose, and dry sticks for arms.
"You are the friendliest snowman in town," Liam whispered, wrapping an old scarf around the snowman's neck to keep him company.""",

    """Chloe loved butterflies. She spent hours in the meadow watching yellow and orange butterflies flutter from daisy to clover.
One day, Chloe noticed a strange green caterpillar resting on a milkweed leaf. The caterpillar was chewing slowly on the edge of the leaf.
"Hello, little crawler," Chloe said quietly.
Every day for two weeks, Chloe visited the milkweed plant. Soon, the caterpillar formed a smooth, jade-green chrysalis with tiny gold spots. It hung motionless from the stem.
One morning, Chloe saw the chrysalis split open. A gorgeous monarch butterfly crawled out, slowly drying its black and orange wings in the warm sunshine.
With a gentle flutter, the butterfly took flight and circled around Chloe once before soaring into the blue sky. Chloe marveled at the magic of nature.""",

    """David loved music. Whenever his mother played the piano, David danced around the living room like a spinning top.
For his fifth birthday, David's parents gave him a small wooden xylophone with rainbow-colored bars and two wooden mallets.
Ding, dong, ping, clink! David tapped on the colored keys.
At first, the sounds were random and loud. But David practiced every afternoon. Soon, he learned how to play a sweet, recognizable melody: "Twinkle, Twinkle, Little Star."
His parents applauded and cheered from the sofa. David beamed with pride. He learned that with practice, he could make beautiful music that brought joy to everyone around him.""",

    """Zoe and her mother went to the beach on a warm Saturday morning. The sand was soft and warm between Zoe's toes.
Zoe carried a blue plastic bucket and a yellow shovel down to the edge of the water.
"Let us build the biggest sandcastle on the whole beach," Zoe suggested excitedly.
They scooped damp sand and built thick, sturdy walls. Zoe collected smooth white seashells and little purple pebbles to decorate the castle gates.
When a gentle ocean wave washed near the castle, Zoe quickly dug a moat around the walls to protect it.
"Look, Mom! The moat kept the castle safe from the wave!" Zoe laughed happily.
Before leaving, Zoe took a photograph with her sandcastle, proud of her architectural masterpiece.""",

    """On a windy autumn afternoon, Toby wanted to fly his diamond-shaped kite. The kite was painted like a smiling green dragon with a long tail of colorful ribbons.
Toby ran across the open meadow as fast as he could.
"Fly, dragon, fly!" shouted Toby.
A strong gust of wind caught the kite and lifted it high into the crisp autumn air. The colorful ribbons danced and fluttered against the clouds.
Suddenly, the wind tugged hard, and the string slipped from Toby's fingers!
"Oh no, my kite!" Toby cried, running after the trailing line.
His big brother Lucas leapt forward and caught the wooden spool just before it flew into a tall elm tree.
"Here you go, Toby. Hold it with both hands," Lucas said kindly.
Together, they steered the green dragon kite high in the sky until the sunset turned the clouds golden.""",

    """A tiny grey squirrel named Sammy lived in a giant hollow maple tree. Autumn had arrived, and the leaves were turning orange, red, and yellow.
Sammy had an important job to do: gather nuts before the cold winter snow arrived.
He scurried across the forest floor, searching beneath the fallen leaves. He found a fat brown acorn!
Sammy dug a small hole near a mossy rock and buried the acorn carefully.
"One for winter, safe and sound," Sammy squeaked to himself.
All day long, Sammy found acorns, walnuts, and chestnuts, hiding them in secret spots all over the woods.
When the first winter snowflakes began to fall, Sammy curled up inside his warm tree hollow, full and cozy.""",

    """Hannah loved painting smooth river stones. She collected ten flat grey pebbles from the creek near her home.
She washed the stones clean and let them dry in the warm afternoon sun.
With her small paintbrushes, Hannah painted bright designs on each stone: a smiling yellow sun, a red ladybug with black spots, a blue flower, and a green tree.
"What will you do with all those pretty stones, Hannah?" her father asked.
"I am going to hide them along the park walking trail for other people to find," Hannah answered with a cheerful smile.
Later that afternoon, a little boy on the walking trail found Hannah's ladybug stone and shouted with delight. Seeing his happiness made Hannah's heart swell with joy.""",

    """Max and his father decided to pitch a tent in the backyard for their first camping adventure.
They unfolded the blue tent fabric and clicked the long metal poles together.
"Push the pole through the sleeve, Max," said his father.
Max pushed carefully until the tent popped up into a sturdy dome. Inside, they laid down two soft sleeping bags and switched on their flashlights.
When night fell, they roasted sweet white marshmallows over a small grill until they were golden and gooey.
Lying inside the tent with the mesh window open, Max listened to the chirping crickets and watched the blinking stars.
"Camping in the backyard is the best adventure in the world, Dad," Max whispered, drifting off into a peaceful sleep.""",

    """A sweet little puppy named Daisy was afraid of thunderstorms. Whenever the thunder boomed, Daisy trembled and hid behind the living room sofa.
One stormy evening, loud thunder rolled across the dark sky: BOOM!
Little six-year-old Henry saw Daisy shaking in fear. Henry took his favorite soft blue blanket and sat down on the rug beside the sofa.
"Come here, Daisy. You are safe with me," Henry said in a soothing voice.
Daisy crept out and rested her chin on Henry's knee. Henry wrapped the warm blanket around the puppy and gently stroked her floppy ears.
Henry read a storybook aloud to distract her from the rain. Soon, Daisy stopped shaking and let out a gentle sigh, resting comfortably in her friend's arms.""",

    """Nora loved helping her mother in the kitchen. Today, they were baking homemade chocolate chip cookies.
Nora poured the white flour, the golden sugar, and the soft butter into a large mixing bowl.
"Now use the wooden spoon to stir everything together, Nora," her mother instructed.
Nora stirred with all her might until the dough was smooth and thick. Then she poured in a whole cup of sweet chocolate chips.
Using two spoons, Nora scooped round balls of dough onto the shiny baking sheet.
The oven baked the cookies to a perfect golden brown. When they took them out, the chocolate chips were warm and melted.
Nora shared the cookies with her family, savoring the delicious treat she helped make from scratch.""",

    """Ethan had a shiny red scooter with two front wheels. He wore his blue safety helmet and strapped on his knee pads.
"Watch me ride, Mom!" Ethan shouted, pushing off the sidewalk with his right foot.
Gliding down the smooth path was thrilling! But as he approached the corner, a stray pebble got stuck in his front wheel. The scooter wobbled, and Ethan tumbled onto the grass.
Ethan felt his knee sting and a tear rolled down his cheek.
His mother walked over calmly, wiped off the grass, and gave his knee a gentle kiss.
"You are okay, brave Ethan. Falling happens to everyone. Do you want to try again?" she asked.
Ethan stood up, brushed off his pants, and stepped back onto the scooter with determination. He kicked off again, riding smoothly with a proud smile.""",

    """Maya loved animals. She had a little orange goldfish named Bubbles that swam in a clear glass bowl on her bedroom dresser.
Every morning, Maya took the small container of fish food and sprinkled just two tiny flakes onto the water's surface.
"Good morning, Bubbles!" Maya chirped.
Bubbles swam to the top and nibbled the flakes with tiny bubbles popping from his mouth.
Maya made sure the water stayed sparkling clean and placed a tiny ceramic castle in the middle of the bowl for Bubbles to explore.
Taking care of Bubbles taught Maya how important it is to be responsible and gentle with pets.""",
]

def create_tinystories_dataset(output_path="data/tinystories.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Combine and multiply to create a robust ~50KB dataset
    full_text = "\n\n".join(STORIES)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"Created TinyStories dataset at {output_path} with {len(full_text):,} characters across {len(STORIES)} distinct story arcs.")

if __name__ == "__main__":
    create_tinystories_dataset()
